<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class BankStatement extends Model
{
    use HasFactory;

    protected $fillable = [
        'bank_name',
        'account_holder',
        'account_number',
        'account_type',
        'statement_date',
        'closing_balance',
    ];

    /**
     * Relationship to the Transaction records.
     */
    public function transactions()
    {
        return $this->hasMany(Transaction::class , 'bank_statement_id');
    }

    /**
     * Relationship to the VTable records.
     */
    public function vtableRecords()
    {
        return $this->hasMany(VTable::class, 'bank_statement_id');
    }
}
