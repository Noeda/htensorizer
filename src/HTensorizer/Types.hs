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
    Shape2D(..),
    shape2D,
    shapeSize,
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

-- dtype sz loc
--
-- Tensors don't know their own shape, but they know their size.
-- Operations that need a shape have the shape as an argument.
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
-- second (and third if there is one) is source, unless commented otherwise.
--
-- E.g. Dupe tgt src

-- Remember to update HTensorizer.Test with new ops if you add any
data TensorProgram
  = MakeTensorConstant !Tensor !Double
  | MakeTensorUninit !Tensor
  -- identity matrix. The second argument is size. Invariant: size = sqrt (tensor size)
  | MakeTensorEye !Tensor !Int
  | Dupe !Tensor !Tensor
  | AddToTensor !Tensor !Tensor
  -- hadamard product
  | MultiplyToTensor !Tensor !Tensor
  -- MatrixMultiplyToTensor: tgt =   src  *   src2   shape    shape2
  -- tgt assumed to have the shape of src * src2
  | MatrixMultiplyToTensor !Tensor !Tensor !Tensor !Shape2D !Shape2D
  | Seq TensorProgram TensorProgram
  | Return !Tensor
  | Nop
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic)

data Shape2D = Shape2D { rows :: !Int, cols :: !Int }
  deriving (Eq, Ord, Show, Read, Typeable, Data, Generic, NFData)

shape2D :: Int -> Int -> Shape2D
shape2D = Shape2D

shapeSize :: Shape2D -> Int
shapeSize (Shape2D rows cols) = rows * cols

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
