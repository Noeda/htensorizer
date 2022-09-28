module HTensorizer.TensorProgramOptimizations (optimize) where

import Control.Monad.Identity
import Control.Monad.Trans.State.Strict
import Data.Foldable
import qualified Data.Map.Strict as M
import Data.Maybe
import qualified Data.Set as S
import HTensorizer.TensorProgram
import HTensorizer.Types

optimize :: TensorProgram -> TensorProgram
optimize prg =
  let optimized = optimizationRound prg
   in if optimized /= prg
        then optimize optimized
        else optimized
  where
    optimizationRound prg =
      removeUnnecessaryConstants $ constantFold $ removeZeroAdds $ removeUnnecessaryDupes $ removeUnusedCode prg

-- Constant folds
--
-- This turns code like:
--
-- a <- constant 2.0
-- b <- constant 3.0
-- add b a
--
-- into:
--
-- a <- constant 2.0
-- b <- constant 5.0
--
-- The algorithm looks for 'constant' instructions and records what constant
-- value a tensor has. Then, if it encounters an instruction that writes to the
-- vector, it will remove that instruction emits a new constant instruction
-- instead.
constantFold :: TensorProgram -> TensorProgram
constantFold prg =
  evalState (traverseFilterForwards prg go) M.empty
  where
    go piece = do
      constants <- get
      case piece of
        Seq _ _ -> return piece
        MakeTensorConstant tgt constant -> do
          put $ M.insert (tensorLocation tgt) constant constants
          return piece
        Dupe tgt src -> do
          modify $ M.delete (tensorLocation tgt)
          case M.lookup (tensorLocation src) constants of
            Nothing -> return piece
            Just constant -> do
              modify $ M.insert (tensorLocation tgt) constant
              return $ MakeTensorConstant tgt constant
        AddToTensor tgt src -> case (M.lookup (tensorLocation tgt) constants, M.lookup (tensorLocation src) constants) of
          (Just tgt_cons, Just src_cons) -> do
            let summed = tgt_cons + src_cons
            put $ M.insert (tensorLocation tgt) summed constants
            return $ MakeTensorConstant tgt summed
          (Just _, Nothing) -> do
            put $ M.delete (tensorLocation tgt) constants
            return piece
          _ -> return piece
        piece -> do
          let writes = tensorProgramWrites piece
          for_ writes $ \writes ->
            when (writes `M.member` constants) $
              modify $
                M.delete writes
          return piece

-- Removes additions that are just zeros.
--
-- Adding zero to anything doesn't do anything.
--
-- This works by walking forward and keeping track which tensors are zero
-- tensors. If we then see operation that adds that tensor to anything, the
-- operation is removed.
removeZeroAdds :: TensorProgram -> TensorProgram
removeZeroAdds prg =
  evalState (traverseFilterForwards prg go) S.empty
  where
    go piece = do
      zero_tensors <- get
      case piece of
        MakeTensorConstant tgt 0.0 -> do
          put $ S.insert (tensorLocation tgt) zero_tensors
          return piece
        Dupe tgt src
          | tensorLocation src `S.member` zero_tensors -> do
              put $ S.insert (tensorLocation tgt) zero_tensors
              return piece
        AddToTensor _ src
          | tensorLocation src `S.member` zero_tensors ->
              return Nop
        AddToTensor tgt _ -> do
          put $ S.delete (tensorLocation tgt) zero_tensors
          return piece
        _ -> return piece

-- Removes writes that happen before a constant or a dupe.
--
-- Removes code like:
--
-- a <- constant 1.0
-- a <- constant 2.0
--
-- Is turned to:
--
-- a <- constant 2.0
--
-- Or:
--
-- a <- constant 5.0
-- a <- dupe b
--
-- Is turned to:
-- a <- dupe b
removeUnnecessaryConstants :: TensorProgram -> TensorProgram
removeUnnecessaryConstants prg =
  evalState (traverseFilterBackwards prg go) S.empty
  where
    go piece = case piece of
      Seq _ _ -> return piece
      MakeTensorConstant tgt _ -> do
        st <- get
        if tensorLocation tgt `S.member` st
          then return Nop
          else do
            put $ S.insert (tensorLocation tgt) st
            return piece
      Dupe tgt _ -> do
        st <- get
        if tensorLocation tgt `S.member` st
          then return Nop
          else do
            put $ S.insert (tensorLocation tgt) st
            return piece
      piece -> do
        let reads = tensorProgramReads piece
        st <- get
        for_ reads $ \read ->
          when (read `S.member` st) $
            modify $
              S.delete read
        new_st <- get
        let writes = tensorProgramWrites piece
        if any (\write -> write `S.member` new_st) (S.toList writes)
          then do
            put st
            return Nop
          else return piece

-- Removes unnecessary dupes
--
-- If we see code like this:
--
-- b <- zeros
-- a <- dupe b
--
-- Then we'd rather just write:
-- a <- zeros
--
-- This algorithm walks the program from start to finish.
--
-- If we see a dupe, then we record "tensor a was duped from b"
-- If there's no operations on b again, then we rewrite 'a' to be 'b' and
-- remove the dupe instruction.
removeUnnecessaryDupes :: TensorProgram -> TensorProgram
removeUnnecessaryDupes prg =
  let rewritables = execState go M.empty
      -- rewriteables will be Map TensorLocation (TensorLocation, Bool) where
      -- if the bool is false, then there's a dangling dupe. We can rewrite the
      -- dupe away.
      rewrites = M.fromList $ catMaybes $ fmap (\(src, (tgt, not_dangling)) -> if not_dangling then Nothing else Just (src, tgt)) (M.assocs rewritables)
      result = runIdentity $ traverseTensorLocations prg $ \loc ->
        case M.lookup loc rewrites of
          Nothing -> return loc
          Just rewrite -> return rewrite
   in removeSelfDupes result
  where
    go = do
      traverseFilterForwards prg $ \piece -> do
        case piece of
          Seq _ _ -> return piece
          any_piece -> do
            let ops = S.union (tensorProgramReads any_piece) (tensorProgramWrites any_piece)
            dupes <- get
            for_ ops $ \op -> do
              when (M.member op dupes) $
                modify $
                  M.adjust (\(tgt, _) -> (tgt, True)) op
            case any_piece of
              Dupe tgt src -> do
                -- Record that a dupe has been from src to tgt
                modify $ M.insert (tensorLocation src) (tensorLocation tgt, False)
                return (Dupe tgt src)
              _ -> return any_piece

-- Removes dupes that dupe to themselves.
--
-- They are produced as a side effect from dupe removal optimization.
--
-- I.e. instructions of form:
-- a <- dupe a
removeSelfDupes :: TensorProgram -> TensorProgram
removeSelfDupes prg = runIdentity $ traverseFilterForwards prg $ \piece -> do
  case piece of
    Dupe src1 src2 | tensorLocation src1 == tensorLocation src2 -> return Nop
    _ -> return piece

-- Removes operations on tensors that are not contributing to the final result.
--
-- The algorithm starts from the end and goes backwards. It marks all tensors
-- that are involved in an operation to the final tensor. If we see any
-- operation that has no effect on tensors we've already marked, then we
-- surmise that piece is unused.
--
-- Other optimizations may expose more code to be removed.
removeUnusedCode :: TensorProgram -> TensorProgram
removeUnusedCode prg =
  flip evalState S.empty $ do
    traverseFilterBackwards prg $ \piece -> do
      -- TODO: some generic mechanism to check "does this instruction read from
      -- this and write to this". Maybe next time some operation needs to know
      -- this.
      marked <- get
      case piece of
        Seq _ _ -> return piece
        Return tensor -> do
          put $ S.insert (tensorLocation tensor) marked
          return piece
        piece -> do
          let reads = tensorProgramReads piece
              writes = tensorProgramWrites piece
          if S.disjoint marked writes
            then return Nop
            else do
              put $ S.union marked reads
              return piece
