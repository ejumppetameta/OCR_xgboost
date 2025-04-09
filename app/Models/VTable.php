<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class VTable extends Model
{
    use HasFactory;

    // Explicitly set the table name if it's not the plural of the model name.
    protected $table = 'vtable';

    protected $fillable = [
        'bank_statement_id',
        'transaction_date',
        'description',
        'category',
        'debit',
        'credit',
        'balance',
        'type',
        'reference_number',
        'dr_mydebit',
    ];

    public function bankStatement()
    {
        return $this->belongsTo(BankStatement::class);
    }
}
