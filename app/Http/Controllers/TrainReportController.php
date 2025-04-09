<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class TrainReportController extends Controller
{
    /**
     * Display a paginated list of training evaluation reports.
     *
     * @return \Illuminate\View\View
     */
    public function index()
    {
        // Retrieve all reports from model_evaluations, ordered by newest first.
        $reports = DB::table('model_evaluations')
            ->orderBy('created_at', 'desc')
            ->paginate(10);

        return view('train_reports.index', compact('reports'));
    }

    /**
     * Display the specified evaluation report.
     *
     * @param  int  $id
     * @return \Illuminate\View\View
     */
    public function show($id)
    {
        $report = DB::table('model_evaluations')->find($id);

        if (!$report) {
            abort(404, 'Report not found');
        }

        return view('train_reports.show', compact('report'));
    }
}
