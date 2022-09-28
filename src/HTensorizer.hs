{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ViewPatterns #-}

-- NEXT STEPS:

-- 1. Multiplication
-- 2. Releasing tensors when they are no longer needed

{-
 - This is a tensor program optimizer and compiler.
 -
 - It handles some common tensor operations and produces a program that runs
 - it.
 -}

module HTensorizer
  ( TensorProgram (),
    TensorProgramT (),
    TensorProgramI,
    Tensor (),
    NumericType (..),
    nicePrint,
    zeros,
    ones,
    uninit,
    add,
    dupe,
    toTensorProgram,
    toTensorProgramM,
    optimize,
  )
where

import Control.Applicative
import Control.Monad.Identity
import Control.Monad.Trans.State.Strict
import Control.Monad.Trans.Writer.Strict
import Data.Data
import Data.Foldable
import qualified Data.Map.Strict as M
import Data.Maybe
import qualified Data.Set as S
import GHC.Generics

data NumericType = Float32
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

newtype TensorLocation = TensorLocation Int
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

data Tensor = Tensor !NumericType !Int !TensorLocation
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

tensorLocation :: Tensor -> TensorLocation
tensorLocation (Tensor _ _ loc) = loc

data TensorProgram
  = MakeTensorConstant !Tensor !Double
  | MakeTensorUninit !Tensor
  | Dupe !Tensor !Tensor
  | AddToTensor !Tensor !Tensor
  | Seq TensorProgram TensorProgram
  | Return !Tensor
  | Nop
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

-- Returns all tensor locations that are written to by the program
tensorProgramWrites :: TensorProgram -> S.Set TensorLocation
tensorProgramWrites Nop = S.empty
tensorProgramWrites (Dupe tgt _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (AddToTensor tgt _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (Seq r1 r2) = S.union (tensorProgramWrites r1) (tensorProgramWrites r2)
tensorProgramWrites (MakeTensorUninit tens) = S.singleton (tensorLocation tens)
tensorProgramWrites (MakeTensorConstant tens _) = S.singleton (tensorLocation tens)
tensorProgramWrites (Return _) = S.empty

-- Returns all tensor locations that are read from by the program
tensorProgramReads :: TensorProgram -> S.Set TensorLocation
tensorProgramReads Nop = S.empty
tensorProgramReads (Return tens) = S.singleton (tensorLocation tens)
tensorProgramReads (Dupe _ src) = S.singleton (tensorLocation src)
tensorProgramReads (AddToTensor _ src) = S.singleton (tensorLocation src)
tensorProgramReads (Seq r1 r2) = S.union (tensorProgramReads r1) (tensorProgramReads r2)
tensorProgramReads (MakeTensorUninit _) = S.empty
tensorProgramReads (MakeTensorConstant _ _) = S.empty

traverseTensorLocation :: Applicative f => Tensor -> (TensorLocation -> f TensorLocation) -> f Tensor
traverseTensorLocation (Tensor dtype sz loc) action = Tensor dtype sz <$> action loc

traverseTensorLocations :: Applicative f => TensorProgram -> (TensorLocation -> f TensorLocation) -> f TensorProgram
traverseTensorLocations Nop _ = pure Nop
traverseTensorLocations (Dupe tgt src) action =
  Dupe <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (MakeTensorConstant tens cons) action = MakeTensorConstant <$> traverseTensorLocation tens action <*> pure cons
traverseTensorLocations (MakeTensorUninit tens) action = MakeTensorUninit <$> traverseTensorLocation tens action
traverseTensorLocations (AddToTensor tgt src) action = AddToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (Seq r1 r2) action = Seq <$> traverseTensorLocations r1 action <*> traverseTensorLocations r2 action
traverseTensorLocations (Return tens) action = Return <$> traverseTensorLocation tens action

instance Semigroup TensorProgram where
  Nop <> x = x
  x <> Nop = x
  x <> y = Seq x y

instance Monoid TensorProgram where
  mempty = Nop

data TensorProgramBuilder = TensorProgramBuilder
  { tensorProgram :: !TensorProgram,
    nextTensorLocation :: !Int
  }
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

type TensorProgramI = TensorProgramT Identity

newtype TensorProgramT m a = TensorProgramT {unwrapTensorProgramT :: StateT TensorProgramBuilder m a}
  deriving (Functor, Monad, Applicative)

nicePrint :: TensorProgram -> String
nicePrint prg = execWriter $ go prg
  where
    tellTensor (Tensor dtype sz (TensorLocation loc)) =
      tell $ "@" <> show loc <> " : " <> show dtype <> " " <> show sz

    go (Seq prog1 prog2) = go prog1 >> go prog2
    go Nop = tell "nop\n"
    go (Dupe tgt src) = do
      tell "dupe ("
      tellTensor tgt
      tell ") <- ("
      tellTensor src
      tell ")\n"
    go (AddToTensor tensor1 tensor2) = do
      tell "add ("
      tellTensor tensor1
      tell ") <- ("
      tellTensor tensor2
      tell ")\n"
    go (MakeTensorConstant tensor constant) = do
      tell "("
      tellTensor tensor
      tell $ ") <- constant " <> show constant <> "\n"
    go (MakeTensorUninit tensor) = do
      tell "("
      tellTensor tensor
      tell ") <- uninit\n"
    go (Return tensor) = do
      tell "return ("
      tellTensor tensor
      tell ")"

toTensorProgram :: TensorProgramT Identity Tensor -> TensorProgram
toTensorProgram prg =
  let Identity result = toTensorProgramM prg
   in result

toTensorProgramM :: Monad m => TensorProgramT m Tensor -> m TensorProgram
toTensorProgramM (unwrapTensorProgramT -> tensor_program) = do
  (ret, st) <- runStateT tensor_program (TensorProgramBuilder {tensorProgram = mempty, nextTensorLocation = 0})
  let prg = mappend (tensorProgram st) (Return ret)
  return prg

emitInstruction :: Monad m => TensorProgram -> TensorProgramT m ()
emitInstruction instruction = TensorProgramT $ do
  st <- get
  put $ st {tensorProgram = mappend (tensorProgram st) instruction}

newTensorLocation :: Monad m => TensorProgramT m TensorLocation
newTensorLocation = TensorProgramT $ do
  st <- get
  let loc = nextTensorLocation st
  put $ st {nextTensorLocation = loc + 1}
  return $ TensorLocation loc

zeros :: Monad m => NumericType -> Int -> TensorProgramT m Tensor
zeros dtype sz = constant dtype sz 0.0

ones :: Monad m => NumericType -> Int -> TensorProgramT m Tensor
ones dtype sz = constant dtype sz 1.0

constant :: Monad m => NumericType -> Int -> Double -> TensorProgramT m Tensor
constant dtype sz dbl = do
  loc <- newTensorLocation
  let tens = Tensor dtype sz loc
  emitInstruction $ MakeTensorConstant tens dbl
  return tens

uninit :: Monad m => NumericType -> Int -> TensorProgramT m Tensor
uninit dtype sz = do
  loc <- newTensorLocation
  let tens = Tensor dtype sz loc
  emitInstruction $ MakeTensorUninit tens
  return tens

add :: Monad m => Tensor -> Tensor -> TensorProgramT m ()
add dst_tensor src_tensor = do
  emitInstruction $ AddToTensor dst_tensor src_tensor

dupe :: Monad m => Tensor -> TensorProgramT m Tensor
dupe src_tensor@(Tensor dtype sz _) = do
  new_loc <- newTensorLocation
  let tgt_tensor = Tensor dtype sz new_loc
  emitInstruction $ Dupe tgt_tensor src_tensor
  return $ Tensor dtype sz new_loc

--------
-- Optimization stuff
--------
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

traverseFilterForwards :: Monad f => TensorProgram -> (TensorProgram -> f TensorProgram) -> f TensorProgram
traverseFilterForwards prg action = go prg
  where
    go (Seq x1 x2) = do
      result_x1 <- go x1
      result_x2 <- go x2
      return $ result_x1 <> result_x2
    go thing = action thing

traverseFilterBackwards :: Monad f => TensorProgram -> (TensorProgram -> f TensorProgram) -> f TensorProgram
traverseFilterBackwards prg action = go prg
  where
    go (Seq x1 x2) = do
      result_x2 <- go x2
      result_x1 <- go x1
      return $ result_x1 <> result_x2
    go thing = action thing
