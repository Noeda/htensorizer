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

double2Float :: Double -> Float
double2Float = fromRational . toRational

emptyInterpreterState :: InterpreterState
emptyInterpreterState = InterpreterState {tensors = M.empty}

makeConstant :: Tensor -> Double -> TensorProgramResult
makeConstant tensor cons = case tensorType tensor of
  Float32 -> Float32Result $ V.replicate (tensorSize tensor) (double2Float cons)

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
        Return src -> do
          old <- lift get
          throwE $ fromJust $ M.lookup (tensorLocation src) (tensors old)
