module Main where

import HTensorizer

main :: IO ()
main = do
  let prg = toTensorProgram prog
  putStrLn $ nicePrint prg
  putStrLn "--- After optimization ---"
  putStrLn $ nicePrint (optimize prg)
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
