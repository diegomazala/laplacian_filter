#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/Exporter.hpp>      // C++ exporter interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing fla

#include "VertexTriangleAdjacency.h"

#define ASSIMP_DOUBLE_PRECISION
typedef ai_real Decimal;






void compute_adjacency(const aiMesh *mesh, std::vector<std::vector<uint32_t>>& vertices_adj)
{
	Assimp::VertexTriangleAdjacency adjacency(mesh->mFaces, mesh->mNumFaces, mesh->mNumVertices);
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
}




void build_laplacian_matrix(
	const std::vector<std::vector<uint32_t>>& vertices_adj,
	const aiVector3D* src_vertices,
	uint32_t number_of_vertices,
	const std::vector<uint32_t> control_indices,
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>& result)
{

	//int iv = 0;
	//for (const std::vector<uint32_t> adj : vertices_adj)
	//{
	//	const aiVector3D& v = src_vertices[iv];
	//	std::cout << std::endl << iv << " : (" << v.x << ' ' << v.y << ' ' << v.z << ") : ";
	//	for (const uint32_t vi : adj)
	//	{
	//		std::cout << ' ' << vi;
	//	}
	//	iv++;
	//}
	//std::cout << std::endl;
	

	const uint32_t Anchors = (uint32_t)control_indices.size();
	const uint32_t N = number_of_vertices;

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> L(N, N);
	L.setZero();

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> delta(N, 3);
	delta.setZero();

	//            -
	//            | d_i     i = j
	// (L_s)_ij = | -1      (i, j) are neighbours
	//            |  0      otherwise
	//            -

	for (uint32_t i = 0; i < N; ++i)
	{
		for (uint32_t j = 0; j < N; ++j)
		{
			if (i == j)
				L(i, j) = (Decimal)(vertices_adj[i].size() - 1);	// di => i=j
			else 
			{
				L(i, j) = (Decimal) 
					(std::find(vertices_adj[i].begin(), vertices_adj[i].end(), j) != vertices_adj[i].end() 
					? -1 : 0);
			}
		}
	}

//	std::cout << std::endl << "L " << std::endl << L << std::endl;
	

	uint32_t i = 0;
	for (const std::vector<uint32_t> adj : vertices_adj)
	{
		const aiVector3D& vi = src_vertices[i];

		aiVector3D sum(0, 0, 0);

		for (const uint32_t vj_ind : adj)
		{
			const aiVector3D& vj = src_vertices[vj_ind];
			sum += (vi - vj);
		}

		aiVector3D d = ((Decimal)1.0 / (adj.size() - 1)) * sum;
		delta(i, 0) = d.x;
		delta(i, 1) = d.y;
		delta(i, 2) = d.z;
		//delta(0, i) = d.x;
		//delta(1, i) = d.y;
		//delta(2, i) = d.z;
		i++;
	}


//	std::cout << std::endl << "delta " << std::endl << delta << std::endl;

	// adding two anchors
	//L(N, 0) = 1;
	//L(N + 1, N - 1) = 1;
	

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> A(N + Anchors, N);
	A.setZero();
	A.block(0, 0, L.rows(), L.cols()) = L;

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> b(N + Anchors, 3);
	b.setZero();
	b.block(0, 0, delta.rows(), delta.cols()) = delta;

	//std::cout << std::endl << "b " << std::endl << b << std::endl;
	//std::cout << std::endl << "adding anchors ..." << std::endl;

	for (i = 0; i < Anchors; ++i)
	{
		uint32_t ctrl_ind = control_indices[i];
		
		A(N + i, ctrl_ind) = 1;
		b.row(N + i) = delta.row(ctrl_ind);
	}

	//A(N, 0) = 1;
	//A(N + 1, 3) = 1;
	//A(N + 2, N - 1) = 1;		
	//b.row(N + 0) = delta.row(0);
	//b.row(N + 1) = delta.row(3);
	//b.row(N + 2) = delta.row(N - 1);
	


	//std::cout << std::endl << "A " << std::endl << A << std::endl;
	//std::cout << std::endl << "b " << std::endl << b << std::endl;
	//std::cout << "A size: " << A.rows() << ' ' << A.cols() << std::endl;
	//std::cout << "b size: " << b.rows() << ' ' << b.cols() << std::endl;

	//result = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
	//std::cout << std::endl << "x " << std::endl << result << std::endl;





	// Solving:
	Eigen::SparseMatrix<Decimal> sparse_A(A.sparseView());
	//std::cout << std::endl << "sparse_A " << std::endl << sparse_A << std::endl;
	//std::cout << "sparse A size: " << sparse_A.rows() << ' ' << sparse_A.cols() << std::endl;
	
	//Eigen::SparseLU<Eigen::SparseMatrix<Decimal>, Eigen::COLAMDOrdering<int> >   solver;
	Eigen::SparseQR<Eigen::SparseMatrix<Decimal>, Eigen::COLAMDOrdering<int> >   solver;
	solver.analyzePattern(sparse_A);
	solver.factorize(sparse_A);
	//Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> x = solver.solve(b);

	result = solver.solve(b);

	if (solver.info() != Eigen::Success) 
		std::cout << "Solver Failed!" << std::endl;
	else
		std::cout << "Solver Success!" << std::endl;

	//std::cout << std::endl << "x sparse " << std::endl << result << std::endl;
	//std::cout << "\n\nSao iguais: " << x.isApprox(result) << std::endl << std::endl;
}



int main(int argc, char* argv[])
{
	std::cout
		<< std::endl
		<< "Usage            : ./<app.exe> <input_model> <output_format> <iterations>" << std::endl
		<< "Default          : ./laplacian_filter.exe ../../data/teddy.obj obj 1" << std::endl
		<< std::endl;

	const std::string input_filename = (argc > 1) ? argv[1] : "../../data/sample_plane.obj";
	const std::string output_format = (argc > 2) ? argv[2] : "obj";

	std::stringstream ss;
	ss << input_filename.substr(0, input_filename.size() - 4)
		<< "_out." << output_format;
	std::string output_filename = ss.str();

	//
	// Import file
	// 
	Assimp::Importer importer;
	const aiScene *scene_in = importer.ReadFile(input_filename, aiProcess_JoinIdenticalVertices | aiProcess_RemoveComponent);	// aiProcessPreset_TargetRealtime_Fast);
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
		<< "File Loaded      : " << input_filename << std::endl
		<< "Vertices         : " << mesh->mNumVertices << std::endl
		<< "Faces            : " << mesh->mNumFaces << std::endl;


	//
	// Compute vertices adjacency
	//
	std::vector<std::vector<uint32_t>> vertices_adj(mesh->mNumVertices);
	compute_adjacency(mesh, vertices_adj);

	//
	// Build laplacian matrix
	//
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> result;
	std::vector<uint32_t> control_indices = { 0, 1, 3 };
	build_laplacian_matrix(vertices_adj, mesh->mVertices, mesh->mNumVertices, control_indices, result);


	//
	// Copy result to mesh
	//
	for (uint32_t i = 0; i < mesh->mNumVertices; ++i)
	{
		mesh->mVertices[i].x = result(i, 0);
		mesh->mVertices[i].y = result(i, 1);
		mesh->mVertices[i].z = result(i, 2);
	}

	Assimp::Exporter exporter;
	aiReturn ret = exporter.Export(scene, output_format, output_filename, scene->mFlags);
	if (ret == aiReturn_SUCCESS)
		std::cout << "Output File      : " << output_filename << std::endl;
	else
		std::cout << "Output File      : <ERROR> file not saved - " << output_filename << std::endl;



	aiFreeScene(scene);

	return EXIT_SUCCESS;
}