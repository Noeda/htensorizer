{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Main (main) where

import HTensorizer.TensorProgram
import HTensorizer.Test
import HTensorizer.Types
import Test.Hspec
import Test.QuickCheck

newtype PrettyPrintedTensorProgram = PrettyPrintedTensorProgram TensorProgram
  deriving (Arbitrary)

instance Show PrettyPrintedTensorProgram where
  show (PrettyPrintedTensorProgram program) = nicePrint program

main :: IO ()
main = hspec $ do
  describe "HTensorizer tests" $ do
    it "Programs generated by Arbitrary.TensorProgram are valid" $
      property $
        \(PrettyPrintedTensorProgram program) -> validCheckPassed $ validCheck program