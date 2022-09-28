{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Main (main) where

import Control.DeepSeq
import HTensorizer.HaskellInterpreter
import HTensorizer.TensorProgram
import HTensorizer.TensorProgramOptimizations
import HTensorizer.Test
import HTensorizer.Types
import Test.Hspec
import Test.QuickCheck

newtype PrettyPrintedTensorProgram = PrettyPrintedTensorProgram TensorProgram
  deriving (Arbitrary)

instance Show PrettyPrintedTensorProgram where
  show (PrettyPrintedTensorProgram program) = nicePrint program <> "\n--- OPTIMIZED ---\n" <> nicePrint (optimize program)

main :: IO ()
main = hspec $ do
  describe "HTensorizer tests" $ do
    it "Programs generated by Arbitrary.TensorProgram are valid" $
      withMaxSuccess 10000 $
        property $
          \(PrettyPrintedTensorProgram program) -> validCheckPassed $ validCheck program

    it "Valid programs do not become invalid if optimized" $
      withMaxSuccess 10000 $
        property $
          \(PrettyPrintedTensorProgram program) -> validCheckPassed $ validCheck (optimize program)

    it "Valid programs can be interpreted without any crashes" $
      withMaxSuccess 10000 $
        property $
          \(PrettyPrintedTensorProgram program) ->
            let result = run program
             in result `deepseq` True

    it "Results from non-optimized and optimized programs are equal" $
      withMaxSuccess 10000 $
        property $
          \(PrettyPrintedTensorProgram program) ->
            let result = run program
                result2 = run (optimize program)
             in result == result2
