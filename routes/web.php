<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\PdfOcrController;
use App\Http\Controllers\PdfOcrControllerCreditSense;
use App\Http\Controllers\PdfOcrControllerMaybank;
use App\Http\Controllers\CategoryController;
use App\Http\Controllers\TrainDataController;
use App\Http\Controllers\StatementController;
use App\Http\Controllers\CsvVtableController;
use App\Http\Controllers\FinancialDataController;

Route::get('/', function () {
    return view('upload'); // returns resources/views/upload.blade.php
});

// Routes for processing PDFs
Route::post('/process-pdf-public', [PdfOcrController::class, 'process'])->name('process-pdf-public');
Route::post('/process-pdf-credit-sense', [PdfOcrControllerCreditSense::class, 'process'])->name('process-pdf-credit-sense');
Route::post('/process-pdf-maybank', [PdfOcrControllerMaybank::class, 'process'])->name('process-pdf-maybank');

Route::post('/csv-statement', [CsvVtableController::class, 'process'])->name('process-csv');


// API endpoints for category management (accessible under /api/categories)
Route::prefix('api')->group(function () {
    Route::get('/categories', [CategoryController::class, 'index']);
    Route::get('/categories/{id}', [CategoryController::class, 'show']);
    Route::post('/categories', [CategoryController::class, 'store']);
    Route::put('/categories/{id}', [CategoryController::class, 'update']);
    Route::delete('/categories/{id}', [CategoryController::class, 'destroy']);
});

// View route for the category management interface (Blade view)
Route::get('/categories', function () {
    return view('categories');
});

Route::get('/statement-details/{id}', [StatementController::class, 'show'])->name('statement.details');

Route::get('/statement/{id}', [StatementController::class, 'show'])->name('statement.show');
Route::get('/statement/{id}/download/pdf', [StatementController::class, 'downloadPdf'])->name('download.pdf');
Route::get('/statement/{id}/download/csv', [StatementController::class, 'downloadCsv'])->name('download.csv');

Route::get('/train-data/upload', [TrainDataController::class, 'index'])->name('train_data.index');
Route::post('/train-data/upload', [TrainDataController::class, 'upload'])->name('train_data.upload');


Route::get('/income', [FinancialDataController::class, 'incomeData']);
Route::get('/expenses', [FinancialDataController::class, 'consumerExpenseData']);


use App\Http\Controllers\TrainReportController;

Route::get('/train-reports', [TrainReportController::class, 'index'])->name('train_reports.index');
Route::get('/train-reports/{id}', [TrainReportController::class, 'show'])->name('train_reports.show');
