<?php

namespace App\Http\Controllers;

use App\Models\Category;
use Illuminate\Http\Request;

class CategoryController extends Controller
{
    // List all categories.
    public function index()
    {
        try {
            $categories = Category::all();
            return response()->json([
                'success' => true,
                'data'    => $categories,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Error fetching categories: ' . $e->getMessage(),
            ], 500);
        }
    }

    // Retrieve a single category by its id.
    public function show($id)
    {
        try {
            $category = Category::findOrFail($id);
            return response()->json([
                'success' => true,
                'data'    => $category,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Category not found: ' . $e->getMessage(),
            ], 404);
        }
    }

    // Insert a new category.
    public function store(Request $request)
    {
        $validated = $request->validate([
            'name'     => 'required|string|unique:categories,name',
            'keywords' => 'required|array',
            'type'     => 'required|string'
        ]);

        try {
            $category = Category::create($validated);
            return response()->json([
                'success' => true,
                'message' => 'Category created successfully',
                'data'    => $category,
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Error creating category: ' . $e->getMessage(),
            ], 500);
        }
    }

    // Update an existing category.
    public function update(Request $request, $id)
    {
        try {
            $category = Category::findOrFail($id);
            $validated = $request->validate([
                'name'     => 'sometimes|required|string|unique:categories,name,' . $category->id,
                'keywords' => 'sometimes|required|array',
                'type'     => 'sometimes|required|string'
            ]);
            $category->update($validated);
            return response()->json([
                'success' => true,
                'message' => 'Category updated successfully',
                'data'    => $category,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Error updating category: ' . $e->getMessage(),
            ], 500);
        }
    }

    // Delete a category.
    public function destroy($id)
    {
        try {
            $category = Category::findOrFail($id);
            $category->delete();
            return response()->json([
                'success' => true,
                'message' => 'Category deleted successfully',
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Error deleting category: ' . $e->getMessage(),
            ], 500);
        }
    }
}
