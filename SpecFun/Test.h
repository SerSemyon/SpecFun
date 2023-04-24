#pragma once

unsigned int FindExecutionTime(void method());
void TestBesselCPU();
//void TestBesselNew();
void TestNeumannCPU();
void TestJ0(); 
void TestJ1();
void TestY0();
void TestY1();
void TestBessel_CUDA();
void TestJ0_CUDA();
void TestJ1_CUDA();
void TestY0_CUDA();
void TestChebyshevPolynomials();