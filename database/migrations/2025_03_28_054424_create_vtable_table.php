<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up()
    {
        Schema::create('vtable', function (Blueprint $table) {
            $table->id();
            $table->foreignId('bank_statement_id')->constrained()->onDelete('cascade');

            // Common fields
            $table->date('transaction_date');
            $table->string('description');
            $table->string('category')->nullable(); // From CreditSense
            $table->decimal('debit', 10, 2)->default(0);
            $table->decimal('credit', 10, 2)->default(0);
            $table->decimal('balance', 15, 2)->default(0);

            // New column for transaction type (CR for credit, DR for debit)
            $table->string('type')->nullable();

            // Fields from original PdfOcrController
            $table->string('reference_number')->nullable();
            $table->string('dr_mydebit')->nullable();

            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::table('vtable', function (Blueprint $table) {
            $table->dropForeign(['bank_statement_id']);
        });
        Schema::dropIfExists('vtable');
    }
};
