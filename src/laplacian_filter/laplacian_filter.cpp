#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/Exporter.hpp>      // C++ exporter interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing fla

#include "VertexTriangleAdjacency.h"


#define ASSIMP_DOUBLE_PRECISION
typedef ai_real Decimal;




void laplacian_filter(
	const std::vector<std::vector<uint32_t>>& vertices_adj,
	const aiVector3D* src_vertices,
	aiVector3D* dst_vertices,
	uint32_t number_of_vertices)
{
	int i = 0;
	for (const std::vector<uint32_t> adj : vertices_adj)
	{
		aiVector3D& out_vert = dst_vertices[i];
		aiVector3D d(0, 0, 0);
		for (const uint32_t vi : adj)
			d += src_vertices[vi]; // sum of adjacent vertices (all inclusive)
		
		out_vert = d * ((ai_real)1.0 / adj.size());		// = d / number_of_adj_verts
		i++;
	}
}


void hc_filter(
	const std::vector<std::vector<uint32_t>>& vertices_adj,
	const aiVector3D* src_vertices, 
	aiVector3D* dst_vertices,
	uint32_t number_of_vertices,
	ai_real alpha, 
	ai_real beta)
{
	//
	// Computing laplacian filter
	// 
	aiVector3D* laplacian_verts = new aiVector3D[number_of_vertices];
	laplacian_filter(vertices_adj, src_vertices, laplacian_verts, number_of_vertices);

	//
	// Computing diff
	// 
	aiVector3D* diff_verts = new aiVector3D[number_of_vertices];
	for (uint32_t i = 0; i < number_of_vertices; ++i)
		diff_verts[i] = laplacian_verts[i] - (alpha * src_vertices[i] + ((ai_real)1.0 - alpha) * src_vertices[i]);


	//
	// Computing filter
	// 
	int i = 0;
	for (const std::vector<uint32_t> adj : vertices_adj)
	{
		aiVector3D& out_vert = dst_vertices[i];
		aiVector3D d(0, 0, 0);
		for (const uint32_t vi : adj)
			d += diff_verts[vi]; // sum of adjacent vertices (all inclusive)

		out_vert = laplacian_verts[i] - (beta * diff_verts[i] + (((ai_real)1.0 - beta) * ((ai_real)1.0 / adj.size())) * d);

		i++;
	}
	
	delete[] laplacian_verts;
	delete[] diff_verts;
}



int main(int argc, char* argv[])
{
	std::cout
		<< std::endl
		<< "Usage            : ./<app.exe> <input_model> <output_format> <iterations>" << std::endl
		<< "Default          : ./laplacian_filter.exe ../../data/teddy.obj obj 1" << std::endl
		<< std::endl;

	const int Dimension = 3;
	const std::string input_filename = (argc > 1) ? argv[1] : "../../data/teddy.obj";
	const std::string output_format = (argc > 2) ? argv[2] : "obj";
	const uint32_t iterations = (argc > 3) ? atoi(argv[3]) : 1;
	const bool use_hc_filter = (argc > 4);
	const ai_real alpha = (ai_real)(argc > 5) ? atoi(argv[5]) : 0.0f;
	const ai_real beta = (ai_real)(argc > 6) ? atoi(argv[6]) : 0.5f;
	
	//
	// Composing output file name
	// 
	std::stringstream ss;
	ss << input_filename.substr(0, input_filename.size() - 4)
		<< ((use_hc_filter) ? "_hc_" : "_")
		<< iterations << '.' << output_format;
	std::string output_filename = ss.str();


	//
	// Import file
	// 
	Assimp::Importer importer;
	const aiScene *scene_in = importer.ReadFile(input_filename, aiProcess_JoinIdenticalVertices | aiProcess_RemoveComponent);
		// aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need
	if (scene_in == nullptr)
	{
		std::cout << "Error: Could not read file: " << input_filename << std::endl;
		return EXIT_FAILURE;
	}
	aiScene *scene;
	aiCopyScene(scene_in, &scene);


	//
	// Output info
	// 
	aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
	std::cout
		<< "File             : " << input_filename << std::endl
		<< "Vertices         : " << mesh->mNumVertices << std::endl
		<< "Faces            : " << mesh->mNumFaces << std::endl;


	
	//
	// Compute vertices adjacency
	//
	Assimp::VertexTriangleAdjacency adjacency(mesh->mFaces, mesh->mNumFaces, mesh->mNumVertices);
	std::vector<std::vector<uint32_t>> vertices_adj(mesh->mNumVertices);
	for (uint32_t i = 0; i < mesh->mNumVertices; ++i)
	{
		uint32_t* adj_tris_ptr = adjacency.GetAdjacentTriangles(i);
		uint32_t& adj_tris_count = adjacency.GetNumTrianglesPtr(i);

		for (uint32_t j = 0; j < adj_tris_count; ++j)
		{
			std::vector<uint32_t>& vert_adj_vec = vertices_adj.at(i);
			const aiFace* face = mesh->mFaces + (*adj_tris_ptr);

			for (uint32_t fi = 0; fi < face->mNumIndices; ++fi)
			{
				const uint32_t vertex_index = face->mIndices[fi];
				if (std::find(vert_adj_vec.begin(), vert_adj_vec.end(), vertex_index) == vert_adj_vec.end())
					vert_adj_vec.push_back(vertex_index);
			}

			adj_tris_ptr++;
		}
	}

	//
	// Computing filter
	//
	
	aiVector3D* vertices_output = new aiVector3D[mesh->mNumVertices];
	if (use_hc_filter)
	{
		for (uint32_t it = 0; it < iterations; ++it)
		{
			hc_filter(vertices_adj, mesh->mVertices, vertices_output, mesh->mNumVertices, alpha, beta);
			std::memcpy(mesh->mVertices, vertices_output, sizeof(aiVector3D) * mesh->mNumVertices);
		}
	}
	else
	{
		for (uint32_t it = 0; it < iterations; ++it)
		{
			laplacian_filter(vertices_adj, mesh->mVertices, vertices_output, mesh->mNumVertices);
			std::memcpy(mesh->mVertices, vertices_output, sizeof(aiVector3D) * mesh->mNumVertices);
		}
	}
	delete[] vertices_output;


	Assimp::Exporter exporter;
	aiReturn ret = exporter.Export(scene, output_format, output_filename, scene->mFlags);
	if (ret == aiReturn_SUCCESS)
		std::cout << "Output File      : " << output_filename << std::endl;
	else
		std::cout << "Output File      : <ERROR> file not saved - " << output_filename << std::endl;

	aiFreeScene(scene);
}