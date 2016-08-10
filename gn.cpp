#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <iostream>
#include <vector>

int main( void ) {
	Eigen::Vector4f raw_v;
	raw_v << 0.5, 0.2, -0.8, 1.0;

	std::vector< float > x( 128 );
	std::vector< float > y( 128 );

	for ( int i = 0; i < x.size(); i++ ) {
		x[i] = ( rand() % 256 ) / 256.0f;	
	}
	
	for ( int i = 0; i < y.size(); i++ ) {
		y[i] =
			raw_v[0] * x[i] * x[i] +
			raw_v[1] * x[i] +
			sin( raw_v[2] ) +
			pow( raw_v[3] * x[i], 3.0f );
	}

	Eigen::Vector4f v;
	v << 0.5, 0.4, -0.9, 1.2;

	std::vector< float > J( 128 * 4 );
	Eigen::Matrix4f A;
	std::vector< float > R( 128 );
	Eigen::Vector4f b;
	Eigen::Vector4f dv;

	for ( int m = 0; m < 16; m++ ) {
		// Jacobian : 128 * 4
		for ( int i = 0; i < 128; i++ ) {
			J[i * 4 + 0] = x[i] * x[i]; // 1
			J[i * 4 + 1] = x[i]; // x
			J[i * 4 + 2] = cos( v[2] ); // x^2
			J[i * 4 + 3] = x[i] * x[i] * x[i] * 3.0f * v[3] * v[3]; // x^3
		}

		// Residual : 128x1
		float rme = 0.0f;
		for ( int i = 0; i < R.size(); i++ ) {
			R[i] =
				v[0] * x[i] * x[i] +
				v[1] * x[i] +
				sin( v[2] ) +
				pow( v[3] * x[i], 3.0f )
				- y[i];

			rme += R[i] * R[i];
		}

		// Normal Matrix : J' * J : 4x128 * 128x4
		for ( int i = 0; i < 4; i++ ) {
			for ( int j = 0; j < 4; j++ ) {
				float sum = 0;
				for ( int k = 0; k < 128; k++ ) {
					sum += J[k * 4 + i] * J[k * 4 + j];
				}
				A(i, j) = sum;
			}
		}

		// Levenberg–Marquardt algorithm
		float lamda = 1.0f - m / 15.0f;
		for ( int i = 0; i < 4; i++ ) {
			A(i, i) *= ( 1 + lamda );
		}

		// b : J' * R : 4x128 * 128x1
		for ( int i = 0; i < 4; i++ ) {
			float sum = 0;
			for ( int k = 0; k < 128; k++ ) {
				sum += J[k * 4 + i] * R[k];
			}
			b[i] = sum;
		}

		dv = A.fullPivHouseholderQr().solve( b );

		v -= dv;

		std::cout << m << "=============:" << std::endl;
		std::cout << "A" << std::endl;
		std::cout << A << std::endl;
		std::cout << "b" << std::endl;
		std::cout << b << std::endl;
		std::cout << "dv" << std::endl;
		std::cout << dv << std::endl;
		std::cout << "v" << std::endl;
		std::cout << v << std::endl;
		std::cout << "rme" << std::endl;
		std::cout << rme << std::endl;
		std::cout << std::endl;
	}

	return 0;
}
