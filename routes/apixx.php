<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\CategoryController;

// Group API routes under an "api" prefix if desired.
Route::prefix('api')->group(function () {
    Route::get('/categories', [CategoryController::class, 'index']);
    Route::get('/categories/{id}', [CategoryController::class, 'show']);
    Route::post('/categories', [CategoryController::class, 'store']);
    Route::put('/categories/{id}', [CategoryController::class, 'update']);
    Route::delete('/categories/{id}', [CategoryController::class, 'destroy']);
});

// Other web routes...
Route::get('/categories', function () {
    return view('categories');
});
