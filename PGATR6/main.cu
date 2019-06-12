
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include "timer.h"

#define THREADS 32*16



struct Triangle
{
	int i1, i2, i3;
};

struct Mesh
{
	std::vector<glm::vec3> vertex;
	std::vector<Triangle> triangles;
};


struct Mesh_GPU
{
	glm::vec3* p1;
	glm::vec3* p2;
	glm::vec3* p3;
	glm::mat4* transform;
};



__global__ void meshToMeshGPU(glm::vec3* points, Triangle* triangles, glm::vec3* p1, glm::vec3* p2, glm::vec3* p3,int len)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < len)
	{
		Triangle triangle = triangles[id];
		glm::vec3 p1T = points[triangle.i1];
		glm::vec3 p2T = points[triangle.i2];
		glm::vec3 p3T = points[triangle.i3];
		p1[id] = p1T;
		p2[id] = p2T;
		p3[id] = p3T;
	}
}

__global__ void preprocesTriangle (glm::vec3* p1, glm::vec3* p2, glm::vec3* p3,glm::mat4* transform,int len)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	glm::mat4 translate; //P1 -->origin
	glm::mat4 rot1; //P2 -->zy plane;
	glm::mat4 rot2; //P2 --> >zAxis;
	glm::mat4 rot3; //P3 --> yz plane;
	if (id < len)
	{
		glm::vec4 mP1 = glm::vec4(p1[id],1.0f);
		glm::vec4 mP2 = glm::vec4(p2[id],1.0f);
		glm::vec4 mP3 = glm::vec4(p3[id], 1.0f);
		translate = glm::translate(glm::mat4(1.0f), glm::vec3(-mP1[0], -mP1[1], -mP1[2]));
		mP1 = translate * mP1;
		mP2 = translate * mP2;
		mP3 = translate * mP3;

		float angle1 = atan2(mP2[0], mP2[2]);

		rot1 = glm::rotate(glm::mat4(1.0f), angle1, glm::vec3(0, 1, 0));
		//mP1 = rot1 * mP1; Not affect
		mP2 = rot1 * mP2;
		mP3 = rot1 * mP3;

		float angle2 = atan2(mP2[1], mP2[2]);
		rot2 = glm::rotate(glm::mat4(1.0f), angle2, glm::vec3(1, 0, 0));
		//mP1 = rot2 * mP1;
		mP2 = rot2 * mP2;
		mP3 = rot2 * mP3;

		float angle3 = atan2(mP3[0], mP3[1]);
		rot3 = glm::rotate(glm::mat4(1.0f), angle3, glm::vec3(0, 0, 1));
		//mP1 = rot3 * mP1;
		//mP2 = rot3 * mP2; Not affect
		mP3 = rot3 * mP3;

		//printf("%f,%f,%f\n", angle1, angle2, angle3);
		//p1[id] = mP1; this is always 0;
		p2[id] = mP2;
		p3[id] = mP3;
		transform[id] = rot3 * rot2 * rot1 * translate;


	}
}

__device__ float distanceToSegment(glm::vec3 p,glm::vec2 v, glm::vec2 w)
{
	glm::vec2 line(v[0] - w[0], v[1] - w[1]);
	glm::vec2 p2(p[1], p[2]);
	const float l2 = pow(line[0],2) + pow(line[1], 2);
	const float t = max(0.0f, min(1.0f, glm::dot(p2 - v, w - v) / l2));
	const glm::vec2 projection = v + t * (w - v);  // Projection falls on the segment
	return glm::distance(p,glm::vec3(0.0f,projection[0],projection[1]));
}

__device__ float pointToTriangleDistance(glm::vec3 point,glm::vec3 p2,glm::vec3 p3)
{
	float dY23 = p2[2] - p3[2];
	float dX23 = p2[1] - p3[1];
	float edge23 = (point[1] - p2[1])*dY23 - (point[2] - p2[2])*dX23;

	float dY03 = -p3[2];
	float dX03 = -p3[1];
	float edge03 = (point[1])*dY03 - (point[2])* dX03;
	//printf("%f,%f,%f\n", point[1], point[2], edge03);

	//printf("%f,%f,%f\n", point[1], edge03, edge23);
	if (point[1] > 0.0f && edge03 > 0.0f && edge23 < 0.0f) //Dentro del triangulo
	{
		//printf("Dentro");
		//printf("%f,%f,%f\n", point[1], edge23, edge03);
		return abs(point[0]);
	}
	glm::vec2 p1_2(0, 0);
	glm::vec2 p2_2(p2[1], p2[2]);
	glm::vec2 p3_2(p3[1], p3[2]);
	float d1 = distanceToSegment(point, p1_2, p2_2);
	float d2 = distanceToSegment(point, p1_2, p3_2);
	float d3 = distanceToSegment(point, p2_2, p3_2);
	return min(d1, min(d2, d3));	
}

__global__ void computeDistance(glm::vec3*p, glm::vec3* p1, glm::vec3* p2, glm::vec3* p3, glm::mat4* transform, float* distances,int lenPoints,int lenTriangles)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < lenPoints )
	{
		glm::vec4 p4(p[id], 1.0f);
		glm::vec3 pT = transform[0] * p4;
		float minDist = pointToTriangleDistance(pT, p2[0], p3[0]);
		for (unsigned int i= 1; i < lenTriangles; i++)
		{
			pT = transform[i] * p4;
			minDist = min(minDist, pointToTriangleDistance(pT, p2[i], p3[i]));
		}
		distances[id] = minDist;
	}
	
}

__global__ void parallelMax(float* v, int len)
{
	__shared__ float shared[THREADS * 2];

	int threadID = threadIdx.x + blockDim.x * 2 * blockIdx.x;
	if (threadID > len)
	{
		shared[threadIdx.x] = v[0];
	} else
	{
		shared[threadIdx.x] = v[threadID];

		if (threadID + blockDim.x > len)
		{
			shared[threadIdx.x + blockDim.x] = v[0];
		} else
		{
			shared[threadIdx.x + blockDim.x] = v[threadID + blockDim.x];
		}
	}

	__syncthreads();

	for (unsigned int desp = blockDim.x; desp > 0; desp /=2)
	{
		if (threadIdx.x < desp)
		{
			shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + desp]);
		}
		__syncthreads();
	}

	v[blockIdx.x] = shared[0];
}


Mesh readMesh (const std::string& filename)
{
	Mesh mesh;
	float x, y, z;
	int i1, i2, i3;
	std::string token;
	std::ifstream file(filename);
	

	while (file >> token)
	{
		if (token == "v")
		{
			file >> x;
			file >> y;
			file >> z;
			glm::vec3 point(x, y, z);
			mesh.vertex.push_back(point);
		} else if (token == "f")
		{
			file >> i1;
			file >> i2;
			file >> i3;
			Triangle triangle{i1 - 1,i2 - 1,i3 - 1};
			mesh.triangles.push_back(triangle);
		}
	}
	return mesh;


}

float getMax(float* vec_d,int len)
{

	int numBlocks = (len + 1) / (THREADS * 2) + 1;
	parallelMax << <numBlocks, THREADS >> > (vec_d, len);
	cudaDeviceSynchronize();

	while (numBlocks > 1) {
		int numBlocksTemp = numBlocks;
		numBlocks = (numBlocks + 1) / (THREADS * 2) + 1;
		parallelMax<< <numBlocks, THREADS >> > (vec_d, numBlocksTemp);
		cudaDeviceSynchronize();
	}
	float max = 10.0f;
	cudaMemcpy(&max, vec_d, sizeof(float), cudaMemcpyDeviceToHost);
	return max;
	
}

	
int main(int argc, char **argv) {
	std::string file (argv[1]);
	GpuTimer timer;
	timer.Start();
	Mesh mesh1 = readMesh(file);
	glm::vec3* point_d;
	Triangle* triangle_d;
	Mesh_GPU mesh1_d;
	cudaMalloc(&point_d, mesh1.vertex.size() * sizeof(glm::vec3));
	cudaMalloc(&triangle_d, mesh1.triangles.size() * sizeof(Triangle));
	cudaMemcpy(point_d, mesh1.vertex.data(), mesh1.vertex.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(triangle_d, mesh1.triangles.data(), mesh1.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&(mesh1_d.p1), mesh1.triangles.size() * sizeof(glm::vec3));
	cudaMalloc(&(mesh1_d.p2), mesh1.triangles.size() * sizeof(glm::vec3));
	cudaMalloc(&(mesh1_d.p3), mesh1.triangles.size() * sizeof(glm::vec3));
	cudaMalloc(&(mesh1_d.transform), mesh1.triangles.size() * sizeof(glm::mat4));

	int numBlocks = (mesh1.triangles.size() + 1) / (THREADS) + 1;
	meshToMeshGPU << <numBlocks, THREADS >> > (point_d, triangle_d, mesh1_d.p1, mesh1_d.p2, mesh1_d.p3, mesh1.triangles.size());
	cudaDeviceSynchronize();
	cudaFree(point_d);
	cudaFree(triangle_d);


	preprocesTriangle << <numBlocks, THREADS >> > (mesh1_d.p1, mesh1_d.p2, mesh1_d.p3, mesh1_d.transform, mesh1.triangles.size());
	Mesh mesh2 = readMesh(argv[2]);
	glm::vec3* vertex2_d;
	cudaMalloc(&vertex2_d, mesh2.vertex.size() * sizeof(glm::vec3));
	cudaMemcpy(vertex2_d, mesh2.vertex.data(), mesh2.vertex.size() * sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	float* distance_d;
	cudaMalloc(&distance_d, mesh2.vertex.size() * sizeof(float));
	numBlocks = (mesh2.vertex.size() + 1) / (THREADS) + 1;
	computeDistance << <numBlocks, THREADS >> > (vertex2_d, mesh1_d.p1, mesh1_d.p2, mesh1_d.p3, mesh1_d.transform, distance_d, mesh2.vertex.size(), mesh1.triangles.size());
	cudaDeviceSynchronize();
	float maxDist = getMax(distance_d, mesh2.vertex.size());
	timer.Stop();
	printf("Hausdorff Dist is: %f\n", maxDist);
	printf("Your code ran in: %f msecs.\n", timer.Elapsed());
  return 0;
}
