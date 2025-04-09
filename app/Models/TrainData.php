<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class TrainData extends Model
{
    use HasFactory;

    protected $table = 'train_data';

    protected $fillable = [
        'description',
        'category',
        'type',
    ];
}
