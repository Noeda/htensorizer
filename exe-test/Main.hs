{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Main (main) where

import Control.DeepSeq
import qualified Data.Vector.Unboxed as V
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

-- tool to run both unoptimized and optimized versions of a program
run2 :: TensorProgram -> (TensorProgramResult, TensorProgramResult)
run2 prg = (run prg, run (optimize prg))

main :: IO ()
main = hspec $ do
  describe "HTensorizer unit tests" $ do
    it "Basic 4 + 3 = 7 works" $ do
      run2
        ( toTensorProgram
            ( do
                a <- constant Float32 1 4.0
                b <- constant Float32 1 3.0
                add a b
                return a
            )
        )
        == (Float32Result (V.singleton 7.0), Float32Result (V.singleton 7.0))

    it "Basic 4 * 3 = 12 works" $ do
      run2
        ( toTensorProgram
            ( do
                a <- constant Float32 1 4.0
                b <- constant Float32 1 3.0
                mult a b
                return a
            )
        )
        == (Float32Result (V.singleton 12.0), Float32Result (V.singleton 12))

  describe "HTensorizer matrix multiplication validation unit tests" $ do
    it "128x128 * 128x128 = 128x128 is accepted" $ do
      let prg = toTensorProgram $ do
                  a <- constant Float32 (128*128) 1.0
                  b <- constant Float32 (128*128) 1.0
                  c <- uninit Float32 (128*128)
                  matMult c a b (shape2D 128 128) (shape2D 128 128)
                  return c
       in validCheckPassed $ validCheck prg

    it "128x128 * 128x128 = 127x127 is NOT accepted" $ do
      let prg = toTensorProgram $ do
                  a <- constant Float32 (128*128) 1.0
                  b <- constant Float32 (128*128) 1.0
                  c <- uninit Float32 (127*127)
                  matMult c a b (shape2D 128 128) (shape2D 128 128)
                  return c
       in not $ validCheckPassed $ validCheck prg

    it "32x15 * 15x99 = 32x99 is accepted" $ do
      let prg = toTensorProgram $ do
                  a <- constant Float32 (32*15) 1.0
                  b <- constant Float32 (15*99) 1.0
                  c <- uninit Float32 (32*99)
                  matMult c a b (shape2D 32 15) (shape2D 15 99)
                  return c
       in validCheckPassed $ validCheck prg

    it "32x16 * 15x99 = 32x99 is NOT accepted" $ do
      let prg = toTensorProgram $ do
                  a <- constant Float32 (32*16) 1.0
                  b <- constant Float32 (15*99) 1.0
                  c <- uninit Float32 (32*99)
                  matMult c a b (shape2D 32 16) (shape2D 15 99)
                  return c
       in not $ validCheckPassed $ validCheck prg

    it "uninitialized sources are not accepted" $ do
      let prg1 = toTensorProgram $ do
                   a <- uninit Float32 (128*128)
                   b <- constant Float32 (128*128) 1.0
                   c <- uninit Float32 (128*128)
                   matMult c a b (shape2D 128 128) (shape2D 128 128)
                   return c
          prg2 = toTensorProgram $ do
                   a <- constant Float32 (128*128) 1.0
                   b <- uninit Float32 (128*128)
                   c <- uninit Float32 (128*128)
                   matMult c a b (shape2D 128 128) (shape2D 128 128)
                   return c
       in not (validCheckPassed $ validCheck prg1) &&
          not (validCheckPassed $ validCheck prg2)

  describe "Matrix transpose unit tests" $ do
    it "Transposing 1x1 matrix does nothing" $ do
      let prg = run2 $ toTensorProgram $ do
                  a <- eye Float32 1
                  b <- uninit Float32 1
                  matTranspose b a (shape2D 1 1)
                  return b
       in prg == (Float32Result (V.singleton 1.0), Float32Result (V.singleton 1.0))

    it "Transposing 20x10 turns to 10x20 matrix" $ do
      let prg = run2 $ toTensorProgram $ do
                  a <- zeros Float32 (20*10)
                  b <- uninit Float32 (10*20)
                  -- Write a 3.5 to (1, 0) so that it would go to (0, 1) after transpose
                  writeRectangle a (shape2D 20 10) 3.5 (rect 1 0 1 1)
                  matTranspose b a (shape2D 20 10)
                  return b
          (Float32Result res1, Float32Result res2) = prg
       in V.length res1 == V.length res2 &&
          V.length res1 == 10*20 &&
          all (\idx -> if idx == 1 then res1 V.! idx == 3.5 && res2 V.! idx == 3.5 else res1 V.! idx == 0 && res2 V.! idx == 0) [0..V.length res1 - 1]

  describe "Matrix multiplication unit tests" $ do
    it "Multiplying identity matrices results in identity matrices" $
      let prg1 = run2 $ toTensorProgram $ do
                   a <- eye Float32 64
                   b <- eye Float32 64
                   c <- eye Float32 64
                   matMult c a b (shape2D 64 64) (shape2D 64 64)
                   return c
          prg2 = run2 $ toTensorProgram $ eye Float32 64
       in prg1 == prg2

    it "Multiplying 2x 2x identity matrices results in 4x identity matrix" $
      let prg1 = run2 $ toTensorProgram $ do
                   a <- eye Float32 64
                   b <- eye Float32 64
                   add a a
                   add b b
                   c <- eye Float32 64
                   matMult c a b (shape2D 64 64) (shape2D 64 64)
                   return c
          prg2 = run2 $ toTensorProgram $ do
                   a <- eye Float32 64
                   add a a
                   add a a
                   return a
       in prg1 == prg2

  describe "HTensorizer optimization tests" $ do
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
