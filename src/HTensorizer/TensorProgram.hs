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
    matTranspose,
    writeRectangle,
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

writeRectangle :: Monad m => Tensor -> Shape2D -> Double -> Rectangle -> TensorProgramT m ()
writeRectangle tgt tgt_shape scalar rect = do
  emitInstruction $ WriteRectangleToTensor tgt tgt_shape scalar rect

mult :: Monad m => Tensor -> Tensor -> TensorProgramT m ()
mult dst_tensor src_tensor = do
  emitInstruction $ MultiplyToTensor dst_tensor src_tensor

matMult :: Monad m => Tensor -> Tensor -> Tensor -> Shape2D -> Shape2D -> TensorProgramT m ()
matMult tgt src1 src2 shape1 shape2 =
  emitInstruction $ MatrixMultiplyToTensor tgt src1 src2 shape1 shape2

matTranspose :: Monad m => Tensor -> Tensor -> Shape2D -> TensorProgramT m ()
matTranspose tgt src shape =
  emitInstruction $ MatrixTransposeToTensor tgt src shape

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
    go (MatrixTransposeToTensor tgt src shape) = do
      tell "transpose ("
      tellTensor tgt
      tell ") <- ("
      tellTensor src
      tell $ ") " <> showShape shape <> "\n"
    go (MakeTensorEye tensor sz) = do
      tell "("
      tellTensor tensor
      tell $ ") <- eye " <> show sz <> "x" <> show sz <> "\n"
    go (Return tensor) = do
      tell "return ("
      tellTensor tensor
      tell ")"
    go (WriteRectangleToTensor tensor shape scalar rect) = do
      tell "write_rectangle_to_tensor ("
      tellTensor tensor
      tell $ ") <- " <> show scalar <> " " <> showShape shape <> " " <> show rect <> "\n"

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

-- Returns all tensor locations that are written to by the program.
-- If the tensor might potentially only be partially written or the result
-- depends onthe contents of the tensor, then also include it in
-- tensorProgramReads.
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
tensorProgramWrites (MatrixTransposeToTensor tgt _ _) = S.singleton (tensorLocation tgt)
tensorProgramWrites (WriteRectangleToTensor _ _ _ rect) | rectSize rect == 0 = S.empty
tensorProgramWrites (WriteRectangleToTensor tgt _ _ _) = S.singleton (tensorLocation tgt)
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
tensorProgramReads (MatrixTransposeToTensor _ src _) = S.singleton (tensorLocation src)
tensorProgramReads (MakeTensorEye _ _) = S.empty
tensorProgramReads (MakeTensorUninit _) = S.empty
tensorProgramReads (MakeTensorConstant _ _) = S.empty
tensorProgramReads (WriteRectangleToTensor tgt _ _ rect) | rectSize rect /= tensorSize tgt = S.singleton (tensorLocation tgt)
tensorProgramReads (WriteRectangleToTensor _ _ _ _) = S.empty

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
traverseTensorLocations (MatrixTransposeToTensor tgt src shape) action =
  MatrixTransposeToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action <*> pure shape
traverseTensorLocations (AddToTensor tgt src) action = AddToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (MultiplyToTensor tgt src) action = MultiplyToTensor <$> traverseTensorLocation tgt action <*> traverseTensorLocation src action
traverseTensorLocations (Seq r1 r2) action = Seq <$> traverseTensorLocations r1 action <*> traverseTensorLocations r2 action
traverseTensorLocations (Return tens) action = Return <$> traverseTensorLocation tens action
traverseTensorLocations (WriteRectangleToTensor tgt shape scalar rect) action =
  WriteRectangleToTensor <$> traverseTensorLocation tgt action <*> pure shape <*> pure scalar <*> pure rect

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
  | NegativeRectangleSize
  | OutOfRangeRectangle
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
--     2c. transpose operation has correct target size
--     2d. rectangle write operations are within range
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
          MatrixTransposeToTensor tgt src shape ->
            typeCheckMatrixTranspose piece tgt src shape
          WriteRectangleToTensor tgt tgt_shape scalar rect ->
            typeCheckWriteRectangle piece tgt tgt_shape scalar rect
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

    assertTensorExists piece tensor errtype = do
      tensor' <- M.lookup (tensorLocation tensor) . tensorTypes <$> get
      when (isNothing tensor') $
        emitValidCheckError piece errtype

    assertNotUninit piece tensor errtype = do
      uninit <- S.member (tensorLocation tensor) . uninitTensors <$> get
      when uninit $
        emitValidCheckError piece errtype

    assertSameTypes piece tgt src = do
      let tgt_type = tensorType tgt
          src_type = tensorType src
      when (tgt_type /= src_type) $
        emitValidCheckError piece IncompatibleTargetAndSourceTensors

    typeCheckWriteRectangle _piece _tgt _tgt_shape _scalar rect | rectSize rect == 0 = return ()
    typeCheckWriteRectangle piece _tgt _tgt_shape _scalar rect | rectSize rect < 0 =
      emitValidCheckError piece NegativeRectangleSize
    typeCheckWriteRectangle piece tgt (Shape2D tgt_rows tgt_cols) _scalar rect = do
      when (tgt_rows * tgt_cols /= tensorSize tgt) $
        emitValidCheckError piece IncompatibleSizes
      let Rectangle rrow rcol rh rw = rect
      when (rrow < 0 || rcol < 0) $
        emitValidCheckError piece OutOfRangeRectangle
      when (rrow + rh > tgt_rows || rcol + rw > tgt_cols) $
        emitValidCheckError piece OutOfRangeRectangle

      -- Don't mark target as initialized if the rectangle doesn't cover entire
      -- area
      when (rrow == 0 && rcol == 0 && rw == tgt_cols && rh == tgt_rows) $
        markTensorInit tgt

    typeCheckMatrixMultiply piece tgt src1 src2 (Shape2D rows1 cols1) (Shape2D rows2 cols2) = do
      -- Check that the shapes of all involved matrices are valid
      when (cols1 /= rows2) $
        emitValidCheckError piece $ IncompatibleSizes
      let tgt_sz = tensorSize tgt
          tgt_shape = Shape2D rows1 cols2
      when (tgt_sz /= shapeSize tgt_shape) $
        emitValidCheckError piece $ IncompatibleSizes

      -- check that all the tensors exist
      assertTensorExists piece tgt TargetTensorDoesNotExist
      assertTensorExists piece src1 SourceTensorDoesNotExist
      assertTensorExists piece src2 SourceTensorDoesNotExist

      assertNotUninit piece src1 SourceTensorUninitialized
      assertNotUninit piece src2 SourceTensorUninitialized

      assertSameTypes piece tgt src1
      assertSameTypes piece tgt src2

      markTensorInit tgt

    typeCheckMatrixTranspose piece tgt src (Shape2D src_rows src_cols) = do
      let tgt_shape = Shape2D src_cols src_rows
      let tgt_sz = tensorSize tgt
      when (tgt_sz /= shapeSize tgt_shape) $
        emitValidCheckError piece $ IncompatibleSizes

      assertTensorExists piece tgt TargetTensorDoesNotExist
      assertTensorExists piece src SourceTensorDoesNotExist
      assertNotUninit piece src SourceTensorUninitialized
      assertSameTypes piece tgt src

    typeCheckCreateTensor piece tensor = do
      old_type <- M.lookup (tensorLocation tensor) . tensorTypes <$> get
      whenJustM old_type $ \old_tensor ->
        typeCheckTensorTypesAndSize piece tensor old_tensor
      markTensorInit tensor
      modify $ \old -> old {tensorTypes = M.insert (tensorLocation tensor) tensor (tensorTypes old)}

    typeCheckBinOpTensor piece tgt src = do
      assertTensorExists piece tgt TargetTensorDoesNotExist
      assertTensorExists piece src SourceTensorDoesNotExist
      assertNotUninit piece src SourceTensorUninitialized
      markTensorInit tgt
      typeCheckTensorTypesAndSize piece tgt src

    typeCheckTensorTypesAndSize piece (Tensor tgt_type tgt_sz _) (Tensor src_type src_sz _) = do
      when (tgt_type /= src_type) $
        emitValidCheckError piece IncompatibleTargetAndSourceTensors
      when (tgt_sz /= src_sz) $
        emitValidCheckError piece IncompatibleSizes

    typeCheckReturnTensor piece src = do
      assertTensorExists piece src SourceTensorDoesNotExist
      old_ret_type <- returnTensor <$> get
      whenJustM old_ret_type $ \old_tensor ->
        typeCheckTensorTypesAndSize piece src old_tensor
      assertNotUninit piece src SourceTensorUninitialized
      modify $ \old -> old {returnTensor = Just src}
