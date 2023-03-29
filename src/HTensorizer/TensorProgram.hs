{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ViewPatterns #-}

module HTensorizer.TensorProgram
  ( toTensorProgram,
    toTensorProgramM,
    emitInstruction,
    nicePrint,
    foldForwards,
    foldBackwards,
    programToList,
    tensorLocationsInProgram,
    traverseFilterForwards,
    traverseFilterBackwards,
    traverseTensorLocation,
    traverseTensorLocations,
    tensorProgramWrites,
    tensorProgramReads,
    ValidCheckResult (..),
    validCheck,
    add,
    mult,
    matMult,
    eye,
    dupe,
    uninit,
    zeros,
    ones,
    constant,
  )
where

import Control.Monad.Identity
import Control.Monad.Trans.State.Strict
import Control.Monad.Trans.Writer.Strict
import Data.Data
import qualified Data.Map.Strict as M
import Data.Maybe
import qualified Data.Sequence as SQ
import qualified Data.Set as S
import GHC.Generics
import HTensorizer.Types

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

mult :: Monad m => Tensor -> Tensor -> TensorProgramT m ()
mult dst_tensor src_tensor = do
  emitInstruction $ MultiplyToTensor dst_tensor src_tensor

matMult :: Monad m => Tensor -> Tensor -> Tensor -> Shape2D -> Shape2D -> TensorProgramT m ()
matMult tgt src1 src2 shape1 shape2 =
  emitInstruction $ MatrixMultiplyToTensor tgt src1 src2 shape1 shape2

-- Creates the identity matrix of NxN size
eye :: Monad m => NumericType -> Int -> TensorProgramT m Tensor
eye dtype sz = do
  loc <- newTensorLocation
  let tens = Tensor dtype (sz*sz) loc
  emitInstruction $ MakeTensorEye tens sz
  return tens

dupe :: Monad m => Tensor -> TensorProgramT m Tensor
dupe src_tensor@(Tensor dtype sz _) = do
  new_loc <- newTensorLocation
  let tgt_tensor = Tensor dtype sz new_loc
  emitInstruction $ Dupe tgt_tensor src_tensor
  return $ Tensor dtype sz new_loc

nicePrint :: TensorProgram -> String
nicePrint prg = execWriter $ go prg
  where
    tellTensor (Tensor dtype sz (TensorLocation loc)) =
      tell $ "@" <> show loc <> " : " <> show dtype <> " " <> show sz

    showShape (Shape2D rows cols) = show rows <> "x" <> show cols

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
    go (MultiplyToTensor tensor1 tensor2) = do
      tell "mult ("
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
    go (MatrixMultiplyToTensor tgt src1 src2 shape1 shape2) = do
      tell "matmult ("
      tellTensor tgt
      tell ") <- ("
      tellTensor src1
      tell ") * ("
      tellTensor src2
      tell $ ") " <> showShape shape1 <> " " <> showShape shape2 <> "\n"
    go (MakeTensorEye tensor sz) = do
      tell "("
      tellTensor tensor
      tell $ ") <- eye " <> show sz <> "x" <> show sz <> "\n"
    go (Return tensor) = do
      tell "return ("
      tellTensor tensor
      tell ")"

programToList :: TensorProgram -> [TensorProgram]
programToList prg =
  execState go []
  where
    go = foldBackwards prg $ \piece -> modify $ \old -> piece : old

foldBackwards :: Applicative f => TensorProgram -> (TensorProgram -> f ()) -> f ()
foldBackwards prg action = go prg
  where
    go (Seq x1 x2) = go x2 *> go x1
    go thing = action thing

foldForwards :: Applicative f => TensorProgram -> (TensorProgram -> f ()) -> f ()
foldForwards prg action = go prg
  where
    go (Seq x1 x2) = go x1 *> go x2
    go thing = action thing

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

-- Returns all tensor locations that are written to by the program
tensorProgramWrites :: TensorProgram -> S.Set TensorLocation
tensorProgramWrites Nop = S.empty
tensorProgramWrites (Dupe tgt _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (AddToTensor tgt _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (MultiplyToTensor tgt _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (Seq r1 r2) = S.union (tensorProgramWrites r1) (tensorProgramWrites r2)
tensorProgramWrites (MakeTensorUninit tens) = S.singleton (tensorLocation tens)
tensorProgramWrites (MakeTensorConstant tens _) = S.singleton (tensorLocation tens)
tensorProgramWrites (MakeTensorEye tens _) = S.singleton (tensorLocation tens)
tensorProgramWrites (MatrixMultiplyToTensor tgt _ _ _ _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (Return _) = S.empty

-- Returns all tensor locations that are read from by the program
tensorProgramReads :: TensorProgram -> S.Set TensorLocation
tensorProgramReads Nop = S.empty
tensorProgramReads (Return tens) = S.singleton (tensorLocation tens)
tensorProgramReads (Dupe _ src) = S.singleton (tensorLocation src)
tensorProgramReads (AddToTensor tgt src) = S.fromList [tensorLocation src, tensorLocation tgt]
tensorProgramReads (MultiplyToTensor tgt src) = S.fromList [tensorLocation src, tensorLocation tgt]
tensorProgramReads (Seq r1 r2) = S.union (tensorProgramReads r1) (tensorProgramReads r2)
tensorProgramReads (MatrixMultiplyToTensor _ src1 src2 _ _) = S.fromList [tensorLocation src1, tensorLocation src2]
tensorProgramReads (MakeTensorEye _ _) = S.empty
tensorProgramReads (MakeTensorUninit _) = S.empty
tensorProgramReads (MakeTensorConstant _ _) = S.empty

traverseTensorLocation :: Applicative f => Tensor -> (TensorLocation -> f TensorLocation) -> f Tensor
traverseTensorLocation (Tensor dtype sz loc) action = Tensor dtype sz <$> action loc

tensorLocationsInProgram :: TensorProgram -> S.Set TensorLocation
tensorLocationsInProgram prg =
  execState go S.empty
  where
    go = traverseTensorLocations prg $ \loc -> do
      modify $ S.insert loc
      return loc

traverseTensorLocations :: Applicative f => TensorProgram -> (TensorLocation -> f TensorLocation) -> f TensorProgram
traverseTensorLocations Nop _ = pure Nop
traverseTensorLocations (Dupe tgt src) action =
  Dupe <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (MakeTensorConstant tens cons) action = MakeTensorConstant <$> traverseTensorLocation tens action <*> pure cons
traverseTensorLocations (MakeTensorUninit tens) action = MakeTensorUninit <$> traverseTensorLocation tens action
traverseTensorLocations (MakeTensorEye tens sz) action = MakeTensorEye <$> traverseTensorLocation tens action <*> pure sz
traverseTensorLocations (MatrixMultiplyToTensor tgt src1 src2 shape1 shape2) action =
  MatrixMultiplyToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src1 action <*> traverseTensorLocation src2 action <*> pure shape1 <*> pure shape2
traverseTensorLocations (AddToTensor tgt src) action = AddToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (MultiplyToTensor tgt src) action = MultiplyToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (Seq r1 r2) action = Seq <$> traverseTensorLocations r1 action <*> traverseTensorLocations r2 action
traverseTensorLocations (Return tens) action = Return <$> traverseTensorLocation tens action

data ValidCheckResult = ValidCheckResult
  { validCheckPassed :: !Bool,
    tensorTypes :: !(M.Map TensorLocation Tensor),
    uninitTensors :: !(S.Set TensorLocation),
    returnTensor :: !(Maybe Tensor),
    validationErrors :: !(SQ.Seq ValidationError)
  }
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

data ValidationError = ValidationError !ValidationErrorType !(Maybe TensorProgram)
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

data ValidationErrorType
  = TargetTensorDoesNotExist
  | SourceTensorDoesNotExist
  | IncompatibleTargetAndSourceTensors
  | IncompatibleSizes
  | SourceTensorUninitialized
  | NoReturn
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

emptyValidCheckResult :: ValidCheckResult
emptyValidCheckResult =
  ValidCheckResult
    { validCheckPassed = True,
      tensorTypes = M.empty,
      uninitTensors = S.empty,
      returnTensor = Nothing,
      validationErrors = SQ.empty
    }

whenJustM :: Monad m => Maybe a -> (a -> m ()) -> m ()
whenJustM (Just thing) action = action thing
whenJustM Nothing _ = return ()

-- Validity checks a tensor program
--
-- This checks that all operations:
--
-- 1. do not operate on uninitialized data
-- 2. all operations take compatible tensor sizes and types
--     2a. operations like basic +, -, * etc. have same shape tensors
--     2b. matrix multiplication result has correct size
validCheck :: TensorProgram -> ValidCheckResult
validCheck prg = execState go emptyValidCheckResult
  where
    go = do
      foldForwards prg $ \piece ->
        case piece of
          Seq _ _ -> error "impossible"
          MakeTensorConstant tensor _ ->
            typeCheckCreateTensor piece tensor
          MakeTensorUninit tensor -> do
            typeCheckCreateTensor piece tensor
            markTensorUninit tensor
          Dupe tgt src -> do
            typeCheckCreateTensor piece tgt
            typeCheckBinOpTensor piece tgt src
          AddToTensor tgt src ->
            typeCheckBinOpTensor piece tgt src
          MultiplyToTensor tgt src ->
            typeCheckBinOpTensor piece tgt src
          MatrixMultiplyToTensor tgt src1 src2 shape1 shape2 ->
            typeCheckMatrixMultiply piece tgt src1 src2 shape1 shape2
          MakeTensorEye tgt sz -> do
            when (tensorSize tgt /= sz*sz) $
              emitValidCheckError piece IncompatibleSizes
            typeCheckCreateTensor piece tgt
          Return src ->
            typeCheckReturnTensor piece src
          Nop -> pure ()
      ret_node <- returnTensor <$> get
      when (isNothing ret_node) $
        emitValidCheckError' NoReturn

    emitValidCheckError' error_type =
      modify $ \old ->
        old
          { validationErrors = validationErrors old SQ.|> ValidationError error_type Nothing,
            validCheckPassed = False
          }

    emitValidCheckError piece error_type =
      modify $ \old ->
        old
          { validationErrors = validationErrors old SQ.|> ValidationError error_type (Just piece),
            validCheckPassed = False
          }

    markTensorUninit tensor = do
      modify $ \old -> old {uninitTensors = S.insert (tensorLocation tensor) (uninitTensors old)}

    markTensorInit tensor = do
      modify $ \old -> old {uninitTensors = S.delete (tensorLocation tensor) (uninitTensors old)}

    typeCheckMatrixMultiply piece tgt src1 src2 (Shape2D rows1 cols1) (Shape2D rows2 cols2) = do
      -- Check that the shapes of all involved matrices are valid
      when (cols1 /= rows2) $
        emitValidCheckError piece $ IncompatibleSizes
      let tgt_sz = tensorSize tgt
          tgt_shape = Shape2D rows1 cols2
      when (tgt_sz /= shapeSize tgt_shape) $
        emitValidCheckError piece $ IncompatibleSizes

      -- check that all the tensors exist
      tgt_old_type <- M.lookup (tensorLocation tgt) . tensorTypes <$> get
      src1_old_type <- M.lookup (tensorLocation src1) . tensorTypes <$> get
      src2_old_type <- M.lookup (tensorLocation src2) . tensorTypes <$> get
      when (isNothing tgt_old_type) $
        emitValidCheckError piece TargetTensorDoesNotExist
      when (isNothing src1_old_type) $
        emitValidCheckError piece SourceTensorDoesNotExist
      when (isNothing src2_old_type) $
        emitValidCheckError piece SourceTensorDoesNotExist

      -- source tensors must not be uninitialized
      src1_uninit <- S.member (tensorLocation src1) . uninitTensors <$> get
      src2_uninit <- S.member (tensorLocation src2) . uninitTensors <$> get
      when (src1_uninit || src2_uninit) $
        emitValidCheckError piece SourceTensorUninitialized

      markTensorInit tgt

      let tgt_type = tensorType tgt
          src1_type = tensorType src1
          src2_type = tensorType src2

      -- types must all be compatible
      when (tgt_type /= src1_type || tgt_type /= src2_type) $
        emitValidCheckError piece IncompatibleTargetAndSourceTensors

    typeCheckCreateTensor piece tensor = do
      old_type <- M.lookup (tensorLocation tensor) . tensorTypes <$> get
      whenJustM old_type $ \old_tensor ->
        typeCheckTensorTypes piece tensor old_tensor
      markTensorInit tensor
      modify $ \old -> old {tensorTypes = M.insert (tensorLocation tensor) tensor (tensorTypes old)}

    typeCheckBinOpTensor piece tgt src = do
      tgt_old_type <- M.lookup (tensorLocation tgt) . tensorTypes <$> get
      src_old_type <- M.lookup (tensorLocation src) . tensorTypes <$> get
      when (isNothing tgt_old_type) $
        emitValidCheckError piece TargetTensorDoesNotExist
      when (isNothing src_old_type) $
        emitValidCheckError piece SourceTensorDoesNotExist
      src_uninit <- S.member (tensorLocation src) . uninitTensors <$> get
      when src_uninit $
        emitValidCheckError piece SourceTensorUninitialized
      markTensorInit tgt
      typeCheckTensorTypes piece tgt src

    typeCheckTensorTypes piece (Tensor tgt_type tgt_sz _) (Tensor src_type src_sz _) = do
      when (tgt_type /= src_type) $
        emitValidCheckError piece IncompatibleTargetAndSourceTensors
      when (tgt_sz /= src_sz) $
        emitValidCheckError piece IncompatibleSizes

    typeCheckReturnTensor piece src = do
      src_old_type <- M.lookup (tensorLocation src) . tensorTypes <$> get
      when (isNothing src_old_type) $
        emitValidCheckError piece SourceTensorDoesNotExist
      old_ret_type <- returnTensor <$> get
      whenJustM old_ret_type $ \old_tensor ->
        typeCheckTensorTypes piece src old_tensor
      src_uninit <- S.member (tensorLocation src) . uninitTensors <$> get
      when src_uninit $
        emitValidCheckError piece SourceTensorUninitialized
      modify $ \old -> old {returnTensor = Just src}
