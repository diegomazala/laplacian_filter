#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

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
	ai_real weight,
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

	Eigen::Matrix<Decimal, Eigen::Dynamic, 1> delta_x(N);
	Eigen::Matrix<Decimal, Eigen::Dynamic, 1> delta_y(N);
	Eigen::Matrix<Decimal, Eigen::Dynamic, 1> delta_z(N);
	delta_x.setZero();
	delta_y.setZero();
	delta_z.setZero();

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
		//aiVector3D d = (Decimal)1.0 / (adj.size()) * (sum - vi);
		//aiVector3D d = sum;
		delta(i, 0) = d.x;
		delta(i, 1) = d.y;
		delta(i, 2) = d.z;

		delta_x(i) = d.x;
		delta_y(i) = d.y;
		delta_z(i) = d.z;

		i++;
	}


	//std::cout << std::endl << "delta " << std::endl << delta << std::endl;

	
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> A(N + Anchors, N);
	A.setZero();
	A.block(0, 0, L.rows(), L.cols()) = L;

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> b(N + Anchors, 3);
	b.setZero();
	//b.block(0, 0, delta.rows(), delta.cols()) = delta;

	//std::cout << std::endl << "b " << std::endl << b << std::endl;
	//std::cout << std::endl << "adding anchors ..." << std::endl;

	// adding anchors
	for (i = 0; i < Anchors; ++i)
	{
		const uint32_t ctrl_ind = control_indices[i];
		
		A(N + i, ctrl_ind) = weight;
		b.row(N + i).x() = weight * src_vertices[ctrl_ind].x;
		b.row(N + i).y() = weight * src_vertices[ctrl_ind].y;
		b.row(N + i).z() = weight * src_vertices[ctrl_ind].z;
	}



	//std::cout << std::endl << "A " << std::endl << A << std::endl;
	//std::cout << std::endl << "b " << std::endl << b << std::endl;
	//std::cout << "A size: " << A.rows() << ' ' << A.cols() << std::endl;
	//std::cout << "b size: " << b.rows() << ' ' << b.cols() << std::endl;

	// Solving:
	Eigen::SparseMatrix<Decimal> sparse_A(A.sparseView());
	
	//Eigen::SimplicialLLT<Eigen::SparseMatrix<Decimal>>   solver(sparse_L);
	//Eigen::SimplicialCholesky<Eigen::SparseMatrix<Decimal>>   solver(sparse_L);
	Eigen::SparseQR<Eigen::SparseMatrix<Decimal>, Eigen::COLAMDOrdering<int> >   solver;
	solver.compute(sparse_A);

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> result_x = solver.solve(b.col(0));
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> result_y = solver.solve(b.col(1));
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> result_z = solver.solve(b.col(2));

	result = Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>::Zero(sparse_A.cols(), 3);

	for (i = 0; i < result.rows(); ++i)
	{
		result(i, 0) = result_x(i);
		result(i, 1) = result_y(i);
		result(i, 2) = result_z(i);
	}

	//std::cout << "\nSolver b xyz \n" << result << std::endl << std::endl;

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
		<< "Usage            : ./<app.exe> <input_model> <percentage_indices> <weight> <output_format>" << std::endl
		<< "Default          : ./laplacian_meshes.exe ../../data/cow.obj 30 100 obj" << std::endl
		<< std::endl;

	const std::string input_filename = (argc > 1) ? argv[1] : "../../data/sample_plane.obj";
	const ai_real percentage_of_control_points = (ai_real)((argc > 2) ? atoi(argv[2]) * 0.01 : 0.3);
	const ai_real weight = (ai_real)((argc > 3) ? atof(argv[3]) : 100);
	const std::string output_format = (argc > 4) ? argv[4] : "obj";

	std::stringstream ss;
	ss << input_filename.substr(0, input_filename.size() - 4)
		<< '_' << (int)(percentage_of_control_points * 100)
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
	// generating control indices
	//
	const uint32_t control_points_count = static_cast<uint32_t>(mesh->mNumVertices * percentage_of_control_points);
	std::vector<uint32_t> control_indices(control_points_count);
	std::cout << "Control Points   : " << control_indices.size() << std::endl;
	//
	// Random control indices
	//
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, mesh->mNumVertices);
	for (uint32_t i = 0; i < control_indices.size(); ++i)
	{
		control_indices[i] = dis(gen);
	}




	//
	// Compute vertices adjacency
	//
	std::vector<std::vector<uint32_t>> vertices_adj(mesh->mNumVertices);
	compute_adjacency(mesh, vertices_adj);

	
	//
	// Build laplacian matrix
	//
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> result;
	


	build_laplacian_matrix(vertices_adj, mesh->mVertices, mesh->mNumVertices, control_indices, weight, result);
	

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