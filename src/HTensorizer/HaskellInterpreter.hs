{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RankNTypes #-}

module HTensorizer.HaskellInterpreter (run, TensorProgramResult (..)) where

import Control.DeepSeq
import Control.Monad.Trans
import Control.Monad.Trans.Except
import Control.Monad.Trans.State.Strict
import Data.Data
import qualified Data.Map.Strict as M
import Data.Maybe
import qualified Data.Vector.Unboxed as V
import GHC.Generics
import HTensorizer.TensorProgram
import HTensorizer.Types

data TensorProgramResult
  = Float32Result (V.Vector Float)
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic, NFData)

data InterpreterState = InterpreterState
  {tensors :: !(M.Map TensorLocation TensorProgramResult)}
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic, NFData)

{-# INLINE zipR #-}
zipR :: (forall a. Num a => a -> a -> a) -> TensorProgramResult -> TensorProgramResult -> TensorProgramResult
zipR action (Float32Result vec1) (Float32Result vec2) = Float32Result $ V.zipWith action vec1 vec2

interpretMatMult :: TensorProgramResult -> TensorProgramResult -> Int -> Int -> Int -> Int -> TensorProgramResult
interpretMatMult (Float32Result src1) (Float32Result src2) rows1 cols1 _rows2 cols2 = Float32Result $
  V.generate tgt_sz $ \idx ->
    let row = idx `div` tgt_cols
        col = idx `mod` tgt_cols
     in V.sum $ V.generate cols1 $ \i ->
          let src1_idx = row * cols1 + i
              src2_idx = i * cols2 + col
           in src1 V.! src1_idx * src2 V.! src2_idx
 where
  tgt_cols = cols2
  tgt_sz = rows1 * cols2

double2Float :: Double -> Float
double2Float = fromRational . toRational

emptyInterpreterState :: InterpreterState
emptyInterpreterState = InterpreterState {tensors = M.empty}

makeConstant :: Tensor -> Double -> TensorProgramResult
makeConstant tensor cons = case tensorType tensor of
  Float32 -> Float32Result $ V.replicate (tensorSize tensor) (double2Float cons)

makeEye :: Tensor -> Int -> TensorProgramResult
makeEye tensor sz = Float32Result $ V.generate (tensorSize tensor) $ \idx ->
  let row = idx `div` sz
      col = idx `mod` sz
   in if row == col then 1 else 0

run :: TensorProgram -> TensorProgramResult
run program =
  case evalState (runExceptT go) emptyInterpreterState of
    Left ret -> ret
    _ -> error "impossible"
  where
    go :: ExceptT TensorProgramResult (State InterpreterState) ()
    go = foldForwards program $ \piece ->
      case piece of
        Seq _ _ -> error "impossible"
        Nop -> return ()
        MakeTensorUninit tensor ->
          lift $ modify $ \old -> old {tensors = M.insert (tensorLocation tensor) (makeConstant tensor (0.0 / 0.0)) (tensors old)}
        MakeTensorConstant tensor cons ->
          lift $ modify $ \old -> old {tensors = M.insert (tensorLocation tensor) (makeConstant tensor cons) (tensors old)}
        MakeTensorEye tensor sz ->
          lift $ modify $ \old -> old {tensors = M.insert (tensorLocation tensor) (makeEye tensor sz) (tensors old)}
        Dupe tgt src ->
          lift $ modify $ \old -> old {tensors = M.insert (tensorLocation tgt) (fromJust $ M.lookup (tensorLocation src) (tensors old)) (tensors old)}
        AddToTensor tgt src ->
          lift $ modify $ \old ->
            old
              { tensors =
                  M.insert
                    (tensorLocation tgt)
                    ( zipR
                        (+)
                        (fromJust $ M.lookup (tensorLocation tgt) (tensors old))
                        (fromJust $ M.lookup (tensorLocation src) (tensors old))
                    )
                    (tensors old)
              }
        MultiplyToTensor tgt src ->
          lift $ modify $ \old ->
            old
              { tensors =
                  M.insert
                    (tensorLocation tgt)
                    ( zipR
                        (*)
                        (fromJust $ M.lookup (tensorLocation tgt) (tensors old))
                        (fromJust $ M.lookup (tensorLocation src) (tensors old))
                    )
                    (tensors old)
              }
        MatrixMultiplyToTensor tgt src1 src2 (Shape2D rows1 cols1) (Shape2D rows2 cols2) ->
          lift $ modify $ \old ->
            old
              { tensors =
                  M.insert
                    (tensorLocation tgt)
                    (interpretMatMult (fromJust $ M.lookup (tensorLocation src1) (tensors old))
                                      (fromJust $ M.lookup (tensorLocation src2) (tensors old))
                                      rows1 cols1
                                      rows2 cols2)
                    (tensors old) }
        Return src -> do
          old <- lift get
          throwE $ fromJust $ M.lookup (tensorLocation src) (tensors old)
