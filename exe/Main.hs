module Main where

import Data.Foldable
import HTensorizer.TensorProgram
import HTensorizer.TensorProgramOptimizations
import HTensorizer.Test
import HTensorizer.Types

main :: IO ()
main = do
  let prg = toTensorProgram prog
  putStrLn $ nicePrint prg
  putStrLn $ show $ validCheck prg
  putStrLn "--- After optimization ---"
  putStrLn $ nicePrint (optimize prg)
  putStrLn $ show $ validCheck (optimize prg)
  progs <- sample' (arbitrary :: Gen TensorProgram)
  for_ progs $ \prog -> do
    putStrLn "--- Test program ---"
    putStrLn $ nicePrint prog
    putStrLn "--- optimized ---"
    putStrLn $ nicePrint (optimize prog)
  where
    prog :: TensorProgramI Tensor
    prog = do
      unused <- zeros Float32 512
      tens <- zeros Float32 256
      tens2 <- ones Float32 256
      tens3 <- dupe tens2
      add tens2 tens2
      add tens3 tens
      add tens3 tens2
      add tens tens2
      return tens3
