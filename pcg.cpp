Eigen::VectorXf Diagonal( const Eigen::MatrixXf &A ) {
	Eigen::VectorXf M(A.cols());
	for ( int i = 0; i < A.cols(); i++ ) {
		float a = A(i, i);
		if ( fabs( a ) > 1e-6 ) {
			M(i) = 1.0f / a;
		} else {
			M(i) = 1.0f;
		}
	}
	return M;
}

Eigen::VectorXf PCG( const Eigen::MatrixXf &A, const Eigen::VectorXf &b, int maxIters ) {
	Eigen::VectorXf M = Diagonal( A );

	Eigen::VectorXf x = Eigen::VectorXf::Zero(A.cols());
	Eigen::VectorXf p = Eigen::VectorXf::Zero(A.cols());
	Eigen::VectorXf r = b; // b - A * x

	for ( int i = 0; i < maxIters; i++ ) {
		Eigen::VectorXf Mr = M.array() * r.array();
		float r_rMr = 1.0f / r.dot( Mr );
		p += Mr * r_rMr;

		Eigen::VectorXf Ap = A * p;
		float r_pAp = 1.0f / p.dot( Ap );
		x +=  p * r_pAp;
		r -= Ap * r_pAp;
	}

	return x;
}