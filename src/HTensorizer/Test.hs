{-# LANGUAGE ViewPatterns #-}

module HTensorizer.Test (module Test.QuickCheck.Arbitrary, module Test.QuickCheck.Gen) where

import Control.Monad
import Control.Monad.Identity
import Control.Monad.Trans
import Control.Monad.Trans.State.Strict
import Data.Foldable
import qualified Data.Map.Strict as M
import Data.Maybe
import qualified Data.Set as S
import Data.Word
import HTensorizer.TensorProgram hiding (createdTensors, uninitTensors)
import HTensorizer.Types
import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Gen

instance Arbitrary NumericType where
  arbitrary = return Float32

data TestGenState = TestGenState
  { testTensorLocations :: M.Map TensorLocation (NumericType, Int),
    createdTensors :: S.Set TensorLocation,
    uninitTensors :: S.Set TensorLocation
  }

testGenState :: [(NumericType, Word32)] -> TestGenState
testGenState types' =
  TestGenState
    { testTensorLocations = M.fromList $ zip (fmap TensorLocation [0 ..]) (fmap (\(x, y) -> (x, fromIntegral y `mod` 10000)) types),
      createdTensors = S.empty,
      uninitTensors = S.empty
    }
  where
    types = replicateList types' 5

-- Utility function that replicates items.
-- E.g. ['a', 'b', 'c'] 2 -> ['a', 'a', 'b', 'b', 'c', 'c']
replicateList :: [a] -> Int -> [a]
replicateList _ 0 = []
replicateList (x : rest) n = replicate n x <> replicateList rest n
replicateList [] _ = []

-- Generates valid programs only
instance Arbitrary TensorProgram where
  shrink Nop = []
  shrink prg =
    let lst = programToList prg
        len = length lst
        -- Try removing tensors and removing all instructions related to them
        tensors_in_prg = tensorLocationsInProgram prg
        prg_candidates =
          catMaybes $
            fmap
              ( \remove_tensor ->
                  let new_prg = runIdentity $ traverseFilterForwards prg $ \piece ->
                        let locs = tensorLocationsInProgram piece
                         in if S.null (S.intersection locs (S.singleton remove_tensor))
                              then return piece
                              else return Nop
                   in if validCheckPassed (validCheck new_prg)
                        then Just new_prg
                        else Nothing
              )
              (S.toList tensors_in_prg)
        -- Try cutting program in half and see if any of them are valid
        half1 = mconcat $ take (len `div` 2) lst
        half2 = mconcat $ drop (len `div` 2) lst
     in prg_candidates
          <> ( if validCheckPassed (validCheck half1)
                 then [half1]
                 else []
             )
          <> ( if validCheckPassed (validCheck half2)
                 then [half2]
                 else []
             )

  arbitrary = do
    -- Generate tensors that might exist in the program
    tensor_shapes <- arbitrary :: Gen [(NumericType, Word32)]

    let empty_tensors = null tensor_shapes

    -- Generate N instructions
    instructions <- arbitrary :: Gen [()]

    let whenNotEmpty action = if empty_tensors then return () else action

    flip evalStateT (testGenState tensor_shapes) $ toTensorProgramM $ do
      let arbw32 = lift $ lift $ arbitrary :: TensorProgramT (StateT TestGenState Gen) Word32
      let arbd64 = lift $ lift $ arbitrary :: TensorProgramT (StateT TestGenState Gen) Double
      let getTensor (fromIntegral -> idx) = do
            st <- lift get
            let loc = TensorLocation (idx `mod` (M.size (testTensorLocations st)))
            let (tp, sz) = fromJust $ M.lookup loc (testTensorLocations st)
            return $ Tensor tp sz loc

          markTensorCreated tensor = lift $ modify $ \old -> old {createdTensors = S.insert (tensorLocation tensor) (createdTensors old)}

          markTensorUninit tensor = lift $ modify $ \old -> old {uninitTensors = S.insert (tensorLocation tensor) (uninitTensors old)}
          markTensorInit tensor = lift $ modify $ \old -> old {uninitTensors = S.delete (tensorLocation tensor) (uninitTensors old)}

          createIfNotCreated tensor = do
            st <- lift get
            let created = S.member (tensorLocation tensor) (createdTensors st)
            let uninited = S.member (tensorLocation tensor) (uninitTensors st)
            unless (created && not uninited) $ do
              cons <- arbd64
              emitInstruction $ MakeTensorConstant tensor cons
            markTensorCreated tensor
            markTensorInit tensor

          tryFindCompatibleTensors action = do
            let attempt tries_left = when (tries_left > 0) $ do
                  src_tensor_idx <- arbw32
                  tgt_tensor_idx <- arbw32
                  src_tensor <- getTensor src_tensor_idx
                  tgt_tensor <- getTensor tgt_tensor_idx
                  if areTensorsCompatible src_tensor tgt_tensor
                    then action tgt_tensor src_tensor
                    else attempt (tries_left - 1 :: Int)
            attempt 10

      for_ instructions $ \() -> do
        -- TODO: should we care about modulo bias? I am too lazy to care
        instruction_op_ <- arbw32
        let instruction_op = instruction_op_ `mod` 5
        src_tensor_idx <- arbw32

        -- TODO: add matrix multiplication

        case instruction_op of
          0 -> do
            -- constant
            whenNotEmpty $ do
              cons <- arbd64
              src_tensor <- getTensor src_tensor_idx
              emitInstruction $ MakeTensorConstant src_tensor cons
              markTensorCreated src_tensor
              markTensorInit src_tensor
          1 -> do
            -- uninit
            whenNotEmpty $ do
              src_tensor <- getTensor src_tensor_idx
              emitInstruction $ MakeTensorUninit src_tensor
              markTensorCreated src_tensor
              markTensorUninit src_tensor
          2 -> do
            -- dupe
            whenNotEmpty $ do
              tryFindCompatibleTensors $ \tgt_tensor src_tensor -> do
                createIfNotCreated src_tensor
                emitInstruction $ Dupe tgt_tensor src_tensor
                markTensorCreated tgt_tensor
                markTensorUninit tgt_tensor
          3 -> do
            -- add-to-tensor
            whenNotEmpty $ do
              tryFindCompatibleTensors $ \tgt_tensor src_tensor -> do
                createIfNotCreated tgt_tensor
                createIfNotCreated src_tensor
                emitInstruction $ AddToTensor tgt_tensor src_tensor
          4 -> do
            -- multiply-to-tensor
            whenNotEmpty $ do
              tryFindCompatibleTensors $ \tgt_tensor src_tensor -> do
                createIfNotCreated tgt_tensor
                createIfNotCreated src_tensor
                emitInstruction $ MultiplyToTensor tgt_tensor src_tensor
          _ -> error "impossible"

      -- Return rensor
      if empty_tensors
        then do constant Float32 1 0.0
        else do
          ret_tensor <- getTensor =<< arbw32
          tensor_created <- S.member (tensorLocation ret_tensor) . createdTensors <$> lift get
          tensor_uninit <- S.member (tensorLocation ret_tensor) . uninitTensors <$> lift get
          when (not tensor_created || tensor_uninit) $ do
            cons <- arbd64
            emitInstruction $ MakeTensorConstant ret_tensor cons
          return ret_tensor
