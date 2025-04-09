<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Category extends Model
{
    protected $fillable = ['name', 'keywords', 'type'];

    // Automatically cast the keywords attribute from JSON to array
    protected $casts = [
        'keywords' => 'array',
    ];

    /**
     * Optional: Define a relationship to transactions.
     */
    public function transactions()
    {
        return $this->hasMany(Transaction::class);
    }

    /**
     * Optional: Ensure that keywords are stored as an array even if provided as a comma-separated string.
     *
     * @param mixed $value
     */
    public function setKeywordsAttribute($value)
    {
        if (is_string($value)) {
            $this->attributes['keywords'] = json_encode(array_filter(array_map('trim', explode(',', $value))));
        } elseif (is_array($value)) {
            $this->attributes['keywords'] = json_encode($value);
        } else {
            $this->attributes['keywords'] = json_encode([]);
        }
    }
}
