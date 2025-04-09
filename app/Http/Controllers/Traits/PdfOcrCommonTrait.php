<?php
namespace App\Http\Controllers\Traits;

use Illuminate\Support\Facades\Log;
use GuzzleHttp\Client;
use App\Models\TrainData;

trait PdfOcrCommonTrait
{
    /**
     * Calls the ML service to get a predicted category.
     *
     * @param string $description
     * @param string $type
     * @return string|null
     */
    protected function getMlCategory($description, $type)
    {
        try {
            $client = new Client(['timeout' => 5]);
            $response = $client->post('http://ml:5000/predict', [
                'json' => [
                    'text' => $description,
                    'type' => $type,
                ]
            ]);
            $result = json_decode($response->getBody(), true);

            if (!is_array($result)) {
                Log::warning("ML service returned non-array response for description: {$description}, type: {$type}");
                return null;
            }

            if (isset($result['category']) && !empty($result['category'])) {
                $confidence = isset($result['confidence'])
                    ? (float)$result['confidence']
                    : (isset($result['category_confidence']) ? (float)$result['category_confidence'] : 1.0);
                if ($confidence >= 0.1) {
                    return $result['category'];
                }
                Log::info("ML prediction confidence too low ({$confidence}) for description: {$description}");
            } else {
                Log::info("ML service did not return a category for description: {$description}");
            }
        } catch (\Exception $e) {
            Log::error("ML service error in getMlCategory: " . $e->getMessage(), [
                'description' => $description,
                'type' => $type
            ]);
        }
        return null;
    }

    /**
     * Uses keyword matching to determine a category.
     *
     * @param string $description
     * @param string $type
     * @param array  $categories
     * @return string|null
     */
    protected function keywordMatchCategory($description, $type, array $categories = [])
    {
        try {
            $description = strtolower($description);
            foreach ($categories as $cat) {
                // Ensure the category structure has the necessary keys.
                if (!isset($cat['type'], $cat['keywords'], $cat['name'])) {
                    Log::warning("Category data structure is missing keys", ['category' => $cat]);
                    continue;
                }

                if (strcasecmp($cat['type'], $type) !== 0) {
                    continue;
                }
                foreach ($cat['keywords'] as $keyword) {
                    // Make sure the keyword is a valid string.
                    if (!is_string($keyword) || empty($keyword)) {
                        continue;
                    }
                    if (preg_match('/\b' . preg_quote(strtolower($keyword), '/') . '\b/', $description)) {
                        return $cat['name'];
                    }
                }
            }
        } catch (\Exception $e) {
            Log::error("Error in keywordMatchCategory: " . $e->getMessage(), [
                'description' => $description,
                'type' => $type,
                'categories' => $categories,
            ]);
        }
        return null;
    }

    /**
     * Determines the final category using both ML and keyword matching.
     *
     * @param string $description
     * @param string $type
     * @param array  $categories
     * @return string
     */
    protected function determineCategory($description, $type, array $categories)
    {
        try {
            $mlCategory = $this->getMlCategory($description, $type);
            if ($mlCategory !== null && $mlCategory !== "Uncertain") {
                return $mlCategory;
            }

            $keywordCategory = $this->keywordMatchCategory($description, $type, $categories);
            if ($keywordCategory !== null) {
                // Train the ML service asynchronously; log any failures.
                $this->trainMlCategory($description, $type, $keywordCategory);
                return $keywordCategory;
            }
        } catch (\Exception $e) {
            Log::error("Error in determineCategory: " . $e->getMessage(), [
                'description' => $description,
                'type' => $type,
                'categories' => $categories,
            ]);
        }
        return 'Uncertain';
    }

    /**
     * Stores training data into the train_data table.
     *
     * @param string $description
     * @param string $type
     * @param string $category
     */
    protected function storeTrainingData($description, $type, $category)
    {
        try {
            TrainData::create([
                'description' => $description,
                'type'        => $type,
                'category'    => $category,
            ]);
        } catch (\Exception $e) {
            Log::error("Error in storeTrainingData: " . $e->getMessage(), [
                'description' => $description,
                'type' => $type,
                'category' => $category,
            ]);
        }
    }

    /**
     * Trains the ML service with the provided category.
     *
     * @param string $description
     * @param string $type
     * @param string $category
     */
    protected function trainMlCategory($description, $type, $category)
    {
        // Store training data regardless of external call outcome.
        $this->storeTrainingData($description, $type, $category);

        try {
            $client = new Client(['timeout' => 5]);
            $client->post('http://ml:5000/train', [
                'json' => [
                    'text'     => $description,
                    'type'     => $type,
                    'category' => $category,
                ]
            ]);
        } catch (\Exception $e) {
            Log::error("ML training error in trainMlCategory: " . $e->getMessage(), [
                'description' => $description,
                'type' => $type,
                'category' => $category,
            ]);
        }
    }

    /**
     * Automatically determines the "uncertain" flag based solely on the provided transaction data.
     *
     * If the environment variable MANUAL_UNCERTAIN_FLAG is set, this function immediately returns that
     * value (cast to int) and bypasses the usual auto-detection. Otherwise, the default logic is executed.
     *
     * @param array $transactionData  The transaction data array (should include 'description' and 'type').
     * @param array $categories       Array of category definitions.
     * @return int
     */
    protected function autoDetermineUncertainFlag(array $transactionData, array $categories)
    {
        if (!is_null(env('MANUAL_UNCERTAIN_FLAG'))) {
            return (int)env('MANUAL_UNCERTAIN_FLAG');
        }

        $description = $transactionData['description'] ?? '';
        $type = $transactionData['type'] ?? '';

        try {
            $finalCategory = $this->determineCategory($description, $type, $categories);
            if ($finalCategory !== "Uncertain") {
                return 0;
            }

            $mlCategory = $this->getMlCategory($description, $type);
            if ($mlCategory !== null && $mlCategory !== "Uncertain") {
                return 1;
            }

            $keywordCategory = $this->keywordMatchCategory($description, $type, $categories);
            if ($keywordCategory !== null) {
                return 2;
            }
        } catch (\Exception $e) {
            Log::error("Error in autoDetermineUncertainFlag: " . $e->getMessage(), [
                'transactionData' => $transactionData,
                'categories' => $categories,
            ]);
        }
        return 0;
    }

    /**
     * Saves transaction data into one or more tables based on the "uncertain" flag.
     *
     * Revised rules:
     *   - If $uncertain equals 0: Save into Transaction.
     *   - If $uncertain equals 1: Save into Transaction and VTable.
     *   - If $uncertain is greater than 1: Save into Transaction, VTable, and optionally store training data.
     *
     * @param \Illuminate\Database\Eloquent\Model $bankStatement The bank statement model instance.
     * @param array $data Array of transaction data (must include keys: description, type, category, etc.).
     * @param int $uncertain The flag that determines the saving strategy.
     * @return void
     */
    protected function saveTransactionData($bankStatement, array $data, $uncertain)
    {
        try {
            // Save into Transaction table.
            $bankStatement->transactions()->create(array_merge($data, ['uncertain_flag' => $uncertain]));

            if ($uncertain === 1) {
                // Save into VTable when flag equals 1.
                $bankStatement->vtableRecords()->create($data);
            } elseif ($uncertain > 1) {
                // Save into VTable and optionally train further when flag is greater than 1.
                $bankStatement->vtableRecords()->create($data);
                // Uncomment below if training data should be stored automatically.
                // $this->storeTrainingData($data['description'], $data['type'], $data['category']);
            }
        } catch (\Exception $e) {
            Log::error("Error in saveTransactionData: " . $e->getMessage(), [
                'data' => $data,
                'uncertain_flag' => $uncertain,
            ]);
        }
    }
}
