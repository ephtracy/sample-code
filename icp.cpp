// created by ephtracy
// sample codes to demonstrate alghorithms of point-to-point/point-to-plane registration

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

const double PI = 3.1415926535897932384626433832795;

// R(x)
Eigen::Matrix3f rot_x_mat( float rad ) {
	float c = cos( rad );
	float s = sin( rad );

	Eigen::Matrix3f m;
	m <<
		1, 0, 0,
		0, c, -s,
		0, s, c
		;

	return m;
}

// R(y)
Eigen::Matrix3f rot_y_mat( float rad ) {
	float c = cos( rad );
	float s = sin( rad );

	Eigen::Matrix3f m;
	m <<
		c, 0, s,
		0, 1, 0,
		-s, 0, c
		;

	return m;
}

// R(z)
Eigen::Matrix3f rot_z_mat( float rad ) {
	float c = cos( rad );
	float s = sin( rad );

	Eigen::Matrix3f m;
	m <<
		c, -s, 0,
		s, c, 0,
		0, 0, 1
		;

	return m;
}

// R(z) * R(y) * R(x)
Eigen::Matrix3f euler_to_mat( const Eigen::Vector3f &euler ) {
	return rot_z_mat( euler.z() ) * rot_y_mat( euler.y() ) * rot_x_mat( euler.x() );
}

// skew( w ) * v = cross( w, v )
// Infinitesimal rotations = skew( w ) + I
Eigen::Matrix3f skew_mat( const Eigen::Vector3f &v ) {
	float x = v.x();
	float y = v.y();
	float z = v.z();
	
	Eigen::Matrix3f sk;
	sk <<
		 0, -z, +y,
		+z,  0, -x,
		-y, +x,  0
	;

	return sk;
}

// E(v)
Eigen::Vector3f mean( const std::vector< Eigen::Vector3f > &points ) {
	if ( points.empty() ) {
		return Eigen::Vector3f::Zero();
	}
	
	Eigen::Vector3f e;
	e.setZero();
	for ( int i = 0; i < points.size(); i++ ) {
		e += points[i];
	}
	e /= (float)( points.size() );

	return e;
}

// sqrt( E( | dst - r * src - t |^2 ) )
float rme( const std::vector< Eigen::Vector3f > &src, const std::vector< Eigen::Vector3f > &dst, const Eigen::Matrix3f &r, const Eigen::Vector3f &t ) {
	if ( src.empty() ) {
		return 0.0f;
	}
	
	float e = 0.0f;
	for ( int i = 0; i < src.size(); i++ ) {
		Eigen::Vector3f d = dst[i] - r * src[i] - t;
		e += d.dot( d );
	}
	e /= (float)( src.size() );

	return sqrt( e );
}

// Point-to-Point ICP
// | R * src + t - dst |^2
float icp_point_to_point_svd(
	const std::vector< Eigen::Vector3f > &src_points,
	const std::vector< Eigen::Vector3f > &dst_points,
	Eigen::Matrix3f &out_r, // rotation
	Eigen::Vector3f &out_t  // translation
) {
	if ( src_points.empty() ) {
		out_r.setIdentity();
		out_t.setZero();
		return 0.0f;
	}
		
	// E(src)
	Eigen::Vector3f src_mean = mean( src_points );

	// E(dst)
	Eigen::Vector3f dst_mean = mean( dst_points );

	// A = Sum( ( src - E(src) ) * ( dst - E(dst) )' )
	Eigen::Matrix3f A;
	A.setZero();
	for ( int i = 0; i < src_points.size(); i++ ) {
		A += ( src_points[i] - src_mean ) * ( dst_points[i] - dst_mean ).transpose();
	}

	// SVD : A = U * D * V'
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// R = V * U'
	out_r = svd.matrixV() * svd.matrixU().transpose();

	// T = E(dst) - R * E(src)
	out_t = dst_mean - out_r * src_mean;

	// rme
	return rme( src_points, dst_points, out_r, out_t );
}

// Point-to-Point ICP
// | R * src + t - dst |^2
float icp_point_to_point_iter(
	const std::vector< Eigen::Vector3f > &src_points,
	const std::vector< Eigen::Vector3f > &dst_points,
	Eigen::Matrix3f &out_r,
	Eigen::Vector3f &out_t 
) {
	if ( src_points.empty() ) {
		out_r.setIdentity();
		out_t.setZero();
		return 0.0f;
	}
		
	Eigen::MatrixXf A( 6, 6 );
	A.setZero();

	Eigen::VectorXf b( 6 );
	b.setZero();

	// A * x = b

	// A = Sum( ( S', I )' * ( S', I ) )
	// x = ( w, t )
	// b = Sum( ( m, d )' )

	// d = dst - src
	// m = cross( src, d )
	A.block( 3, 3, 3, 3 ) = Eigen::Matrix3f::Identity() * src_points.size();
	for ( int i = 0; i < src_points.size(); i++ ) {
		Eigen::Vector3f src = out_r * src_points[i] + out_t;
		Eigen::Vector3f dst = dst_points[i];
			
		Eigen::Matrix3f S = skew_mat( src );
		Eigen::Vector3f d = dst - src;
		Eigen::Vector3f m = src.cross( d );

		A.block( 0, 0, 3, 3 ) +=  S * -S;
		A.block( 0, 3, 3, 3 ) +=  S;
		A.block( 3, 0, 3, 3 ) += -S;

		b.block( 0, 0, 3, 1 ) += m;
		b.block( 3, 0, 3, 1 ) += d;
	}

	// solve A * x = b
	Eigen::VectorXf wt( 6 );
	wt = A.fullPivHouseholderQr().solve( b );

	// rotation
	Eigen::Vector3f euler;
	euler << wt[0], wt[1], wt[2];
	Eigen::Matrix3f r = euler_to_mat( euler );

	// translation
	Eigen::Vector3f t;
	t << wt[3], wt[4], wt[5];

	// update transform
	// dst = r * ( out_r * src + out_t ) + t;
	out_r = r * out_r;
	out_t = r * out_t + t;

	// rme
	return rme( src_points, dst_points, out_r, out_t );
}

// Point-to-Plane ICP
// | ( R * src + t - dst )' * n |^2
float icp_point_to_plane_iter(
	const std::vector< Eigen::Vector3f > &src_points,
	const std::vector< Eigen::Vector3f > &dst_points,
	const std::vector< Eigen::Vector3f > &dst_normals,
	Eigen::Matrix3f &out_r,
	Eigen::Vector3f &out_t 
) {
	if ( src_points.empty() ) {
		out_r.setIdentity();
		out_t.setZero();
		return 0.0f;
	}

	Eigen::MatrixXf A( 6, 6 );
	A.setZero();

	Eigen::VectorXf b( 6 );
	b.setZero();

	// A * x = b

	// A = Sum( ( m, n ) * ( m, n )' )
	// x = ( w, t )
	// b = Sum( ( m, n ) * d )

	// d = dot( ( dst - src ), n )
	// m = cross( src, n )
	for ( int i = 0; i < src_points.size(); i++ ) {
		Eigen::Vector3f src = out_r * src_points[i] + out_t;
		Eigen::Vector3f dst = dst_points[i];
		Eigen::Vector3f n = dst_normals[i];

		float d = ( dst - src ).dot( n );
		Eigen::Vector3f m = src.cross( n );

		Eigen::VectorXf v( 6 );
		v.block( 0, 0, 3, 1 ) = m;
		v.block( 3, 0, 3, 1 ) = n;

		A += v * v.transpose();

		b += v * d;
	}

	// solve A * x = b
	Eigen::VectorXf wt( 6 );
	wt = A.fullPivHouseholderQr().solve( b );

	// rotation
	Eigen::Vector3f euler;
	euler << wt[0], wt[1], wt[2];
	Eigen::Matrix3f r = euler_to_mat( euler );

	// translation
	Eigen::Vector3f t;
	t << wt[3], wt[4], wt[5];

	// update transform
	// dst = r * ( out_r * src + out_t ) + t;
	out_r = r * out_r;
	out_t = r * out_t + t;

	// rme
	return rme( src_points, dst_points, out_r, out_t );
}

int main() {
	// gen sample data
	std::vector< Eigen::Vector3f > src_points;
	std::vector< Eigen::Vector3f > dst_points, dst_normals;

	// dst
	dst_points.resize( 128 * 128 );
	dst_normals.resize( 128 * 128 );
	for ( int y = 0; y < 128; y++ ) {
		for ( int x = 0; x < 128; x++ ) {
			int index = x + y * 128;

			float u = x / 128.0f * PI;
			float v = y / 128.0f * PI;
			dst_points[index] << u, v, u * u * u + v * v; // ( sin(x), cos(y), x^2 + y^2 )

			Eigen::Vector3f s, t;
			s << 1, 0, 3.0f * u * u;
			t << 0, 1, 2.0f * v;

			dst_normals[index] = s.cross( t );

			//std::cout << dst_normals[index] << std::endl;

			dst_normals[index].normalize();
		}
	}

	// transform
	Eigen::Vector3f euler;
	euler << 20, 50, -45;
	Eigen::Matrix3f rot = euler_to_mat( euler * PI / 180.0 );
	Eigen::Vector3f t;
	t << 30, 50, -10;

	// src
	src_points.resize( 128 * 128 );
	for ( int y = 0; y < 128; y++ ) {
		for ( int x = 0; x < 128; x++ ) {
			src_points[x + y * 128] = ( ( dst_points[x + y * 128] - t ).transpose() * rot ).transpose();
		}
	}
	
	// icp : iterative
	if ( 1 ) {
		// init pose
		Eigen::Matrix3f out_rot = Eigen::Matrix3f::Identity();
		Eigen::Vector3f out_t = Eigen::Vector3f::Zero();
	
		// update pose
		for ( int i = 0; i < 15; i++ ) {	
			float rme = icp_point_to_point_iter( src_points, dst_points, out_rot, out_t );

			if ( i < 14 ) {
				continue;
			}
			
			std::cout << ">> Iter " << i << " ======" << std::endl;
			std::cout << out_rot << std::endl;
			std::cout << out_t << std::endl;
			std::cout << "rme : " << rme << std::endl;
			std::cout << std::endl;
			//std::cout << out_rot.determinant() << std::endl;
		}
	}

	// icp : svd
	if ( 1 ) {
		Eigen::Matrix3f out_rot = Eigen::Matrix3f::Identity();
		Eigen::Vector3f out_t = Eigen::Vector3f::Zero();
		
		float rme = icp_point_to_point_svd( src_points, dst_points, out_rot, out_t );
		std::cout << ">> SVD ======" << std::endl;
		std::cout << out_rot << std::endl;
		std::cout << out_t << std::endl;
		std::cout << "rme : " << rme << std::endl;
		std::cout << std::endl;
	}

	// icp : Plane
	{
		// init pose
		Eigen::Matrix3f out_rot = Eigen::Matrix3f::Identity();
		Eigen::Vector3f out_t = Eigen::Vector3f::Zero();
		
		// update pose
		for ( int i = 0; i < 15; i++ ) {	
			float rme = icp_point_to_plane_iter( src_points, dst_points, dst_normals, out_rot, out_t );

			if ( i < 14 ) {
				continue;
			}
			
			std::cout << ">> Plane " << i << " ======" << std::endl;
			std::cout << out_rot << std::endl;
			std::cout << out_t << std::endl;
			std::cout << "rme : " << rme << std::endl;
			std::cout << std::endl;
		}
	}

	// ground truth
	std::cout << ">> Ground Truth =======" << std::endl;
	std::cout << rot << std::endl;
	std::cout << t << std::endl;
}

