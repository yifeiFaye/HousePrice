## multi-level model 
## run the data cleaning part in the "price_predict.py"

## plot the neighborhood agains log price

X3 = X.copy()
X_train = X3.loc[X3['Id'].isin(Id_train), :]
X_test = X3.loc[X3['Id'].isin(Id_test),:]

train = pd.merge(X_train, y_train, left_index = True, right_index = True)

train.group_by('Neighborhood').count()
ax = sns.scatterplot(y = "Neighborhood", x = "logprice",  data = train)
plt.show()

neighbor = X_train.Neighborhood.unique()
n_neighbor = len(neighbor)

neighbor_lookup = dict(zip(neighbor, range(n_neighbor)))
neighbors = X_train['Neighborhood_code'] = X_train.Neighborhood.replace(neighbor_lookup).values 

features = feat_imp[feat_imp > 0.01].index
features = features[features != "Neighborhood"]
n_feature = len(features)

X_train = X_train[features]
X_train1 = pd.get_dummies(X_train[['SaleCondition','Foundation','Condition1',"PavedDrive","ExterCond","HouseStyle", "LotShape"]], drop_first = True)
X_train2 = X_train.drop(columns = ['SaleCondition','Foundation','Condition1',"PavedDrive","ExterCond","HouseStyle", "LotShape"])
X_train = pd.merge(X_train1, X_train2, left_index = True, right_index = True)

features = X_train.columns
n_feature = len(features)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)

hierarchical_intercept = """
data{
	int<lower = 0> N;                //num individuals
	int<lower = 1> K;                //num ind predictors
	int<lower = 1> J;                //num groups
	int<lower = 1> L;                //num group predictors
	int<lower = 1, upper = J> jj[N]; //group for individual
	matrix[N, K] x;                  //individual predictors
	matrix[J, L] u;					 //group predictors
	vector[N] y;                     //outcomes
}
parameters{
	corr_matrix[K] Omega;            //prior covariance
	vector<lower = 0>[K] tau;        //prior scale
	matrix[L, K] gamma;              //group coeffs
	vector[K] beta[J];               //indiv coeffs by group
	real<lower = 0> sigma;           //prediction error scale
}
model{
	matrix[K, K] Sigma_beta;
	Sigma_beta <- diag_matrix(tau) * Omega * dia_matrix(tau);

	tau ~ cauchy(0, 2.5);
	Omega ~ lkj_corr(2);
	
	for (l in 1:L)
		gamma[l] ~ normal(0, 5);

	for (j in 1:J)
		beta[j] ~ multi_normal((u[j] * gamma)', Sigma_beta);

	for (n in 1:N)
		y[n] ~normal(x[n] * beta[jj[n]], sigma);
}