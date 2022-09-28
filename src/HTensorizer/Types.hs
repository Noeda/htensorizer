{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- NEXT STEPS:

-- 1. Multiplication
-- 2. Releasing tensors when they are no longer needed

{-
 - This is a tensor program optimizer and compiler.
 -
 - It handles some common tensor operations and produces a program that runs
 - it.
 -}

module HTensorizer.Types
  ( TensorProgram (..),
    TensorProgramBuilder (..),
    TensorProgramT (..),
    TensorProgramI,
    TensorLocation (..),
    Tensor (..),
    NumericType (..),
    tensorType,
    tensorSize,
    tensorLocation,
    areTensorsCompatible,
  )
where

import Control.Applicative
import Control.DeepSeq
import Control.Monad.Identity
import Control.Monad.Trans
import Control.Monad.Trans.State.Strict
import Data.Data
import Data.Foldable
import Data.Maybe
import GHC.Generics

-- Remember to update HTensorizer.Test with new types if you add any
data NumericType = Float32
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic, NFData)

newtype TensorLocation = TensorLocation Int
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)
  deriving anyclass (NFData)

data Tensor = Tensor !NumericType !Int !TensorLocation
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic, NFData)

tensorType :: Tensor -> NumericType
tensorType (Tensor tp _ _) = tp

tensorSize :: Tensor -> Int
tensorSize (Tensor _ sz _) = sz

tensorLocation :: Tensor -> TensorLocation
tensorLocation (Tensor _ _ loc) = loc

areTensorsCompatible :: Tensor -> Tensor -> Bool
areTensorsCompatible (Tensor tgt_type tgt_sz _) (Tensor src_type src_sz _) =
  tgt_type == src_type && tgt_sz == src_sz

-- when there's an operation with two tensors, the first tensor is target, and
-- second is source, unless commented otherwise
--
-- E.g. Dupe tgt src

-- Remember to update HTensorizer.Test with new ops if you add any
data TensorProgram
  = MakeTensorConstant !Tensor !Double
  | MakeTensorUninit !Tensor
  | Dupe !Tensor !Tensor
  | AddToTensor !Tensor !Tensor
  | MultiplyToTensor !Tensor !Tensor
  | Seq TensorProgram TensorProgram
  | Return !Tensor
  | Nop
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

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
  deriving newtype (Functor, Monad, Applicative)

instance MonadTrans TensorProgramT where
  {-# INLINE lift #-}
  lift action =
    TensorProgramT $
      lift action
