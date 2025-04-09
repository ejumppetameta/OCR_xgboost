// database/migrations/2025_04_09_000000_create_model_evaluations_table.php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateModelEvaluationsTable extends Migration
{
    public function up()
    {
        Schema::create('model_evaluations', function (Blueprint $table) {
            $table->id();
            $table->string('evaluation_type');
            $table->text('report');
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('model_evaluations');
    }
}
