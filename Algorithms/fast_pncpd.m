function [model] =  fast_pncpd(Y, R)
%% Parameters setting 
maxit= 2500;
Y = tensor(Y);
dimY = size(Y);
N = ndims(Y);
epsilon = 1e-3;

%% Initialization 
a_gamma0     = 1e-6;
b_gamma0     = 1e-6;
a_beta0      = 1e-6;
b_beta0      = 1e-6;
gammas = 1*ones(R,1);
beta = 1;
rw = 0.99;
Z = cell(N,1);
ZLambda = cell(N,1);
for n = 1:N
    ZLambda{n} = zeros(R,R);
    [U, S, ~] = svd(double(tenmat(Y,n)), 'econ'); %SVD init 
    if R <= size(U,2)
       Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
    else
        Z{n} = [U*(S.^(0.5))  randn(dimY(n), R-size(U,2))];
    end
end
for n = 1 : N
   ZLambda{n} = zeros(R,R); 
end
EZZT = cell(N,1);
for n=1:N
    EZZT{n} = Z{n}'*Z{n};
end

%% Model learning
X= double(ktensor(Z));
Z0 = Z; % old update
Zm = Z; % extrapolation of Z
t0 = 1; %used for extrapolation weight update
wZ = ones(N,1); % extrapolation weight array
L0 = ones(N,1); L = ones(N,1); % Lipschitz constant array
obj0 = norm(Y)^2;
gammas0 = gammas;
beta0 = beta;
flag_rank_change = 0;
R0 = R;
for it = 1 : maxit
    %% Update factor matrices
    % Z: new updates
    % Z0: old updates
    % Zm: extrapolations of Z
    % L, L0: current and previous Lipschitz bounds
    % obj, obj0: current and previous objective value
    Aw = diag(gammas);
    for n = 1 : N
        % compute E(Z_{\n}^{T} Z_{\n})
        ENZZT = ones(R,R);
        for m = [1:n-1, n+1:N]
            ENZZT =  ENZZT.*EZZT{m};
        end
        % compute E(Z_{\n})
        DDD=double(tenmat(Y, n));
        ZLambda{n} = beta * ENZZT + Aw;
        L0(n) = L(n);
        L(n) = norm(ZLambda{n});
        % fprintf('current time is %f, and the etime is %f \n', time, etime(t2,t1));
        B = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
        FFF = beta * double(DDD*B);        
        Gn = Zm{n}* ZLambda{n}  -  FFF;                
        Z{n} = max(0, Zm{n}- Gn/L(n));
        EZZT{n} = Z{n}'*Z{n};
    end   
    
    %% Update latent tensor X
    X0 = X;
    X = double(ktensor(Z));
    diff = X0 - X;
    if norm(diff(:),'fro')<=1e-3
        break;
    end

    %% Update hyperparameters gamma
    a_gammaN = (sum(dimY(1:N)) + a_gamma0 )*ones(R,1);
    b_gammaN = 0;
    for n=1:N
        b_gammaN = b_gammaN + diag(Z{n}'*Z{n});
    end
    b_gammaN = b_gamma0 + b_gammaN ;
    gammas = (-(b_gammaN - epsilon*gammas) + sqrt((b_gammaN - epsilon*gammas).^2 + 4*epsilon*a_gammaN))/(2*epsilon);

    %% update noise beta
    Y_vec = Y(:);
    X_vec = X(:);
    err = Y_vec'*Y_vec - 2*real(Y_vec'*X_vec) + X_vec'*X_vec;
    a_betaN = a_beta0 + prod(dimY);
    b_betaN = b_beta0 + err;
    beta = (- (b_betaN - epsilon*beta) + sqrt((b_betaN - epsilon*beta)^2 + 4*epsilon*a_betaN ))/(2*epsilon);

    %% diagnostics 
    obj_sum1 = 0.5*prod(dimY)*log(beta) - 0.5*err;
    obj_sum2 = 0.5*sum(dimY)*sum(log(gammas));
    obj_sum3 = 0;
    for n = 1 : N
       obj_sum3 = obj_sum3 - 0.5*trace(Z{n}*diag(gammas)*Z{n}');
    end
    obj_sum4 = sum((a_gamma0 - 1)*log(gammas)-b_gamma0*gammas);
    obj_sum5 = (a_beta0 - 1)*log(beta) - b_beta0*beta;
    obj = -(obj_sum1 + obj_sum2 + obj_sum3 + obj_sum4 + obj_sum5);
  
    t = (1+sqrt(1+4*t0^2))/2;
    if obj > obj0 && flag_rank_change == 0
        Zm = Z0;
        Z = Z0;
        gammas = gammas0;
        beta = beta0;
        for n=1:N
           EZZT{n} = Z{n}'*Z{n};
        end
    else
        % apply extrapolation
        w = (t0-1)/t; % extrapolation weight
        for n = 1 : N
            wZ(n) = min([w,rw*sqrt(L0(n)/L(n))]); % choose smaller weights for convergence
            Zm{n} = Z{n}+wZ(n)*(Z{n}-Z0{n}); % extrapolation
        end
        Z0 = Z; t0 = t; obj0 = obj; gammas0 = gammas; beta0 = beta;
    end
    
    %% ARD
    if it >= 1
        DIMRED_THR=5;
        MAX_GAMMA = min(gammas) * DIMRED_THR;
        if sum(find(gammas > MAX_GAMMA)),
            indices = find(gammas <= MAX_GAMMA);
            for n=1:N
                Z{n} = Z{n}(:,indices);
                Z0{n} = Z0{n}(:,indices);
                Zm{n} = Zm{n}(:,indices);
                ZLambda{n} = ZLambda{n}(indices,indices);
                EZZT{n} = EZZT{n}(indices,indices);
            end
            gammas = gammas(indices);
            R=length(gammas);
        end
    end
    
    if (R ~= R0)
        flag_rank_change = 1;
    else
        flag_rank_change = 0;
    end
    R0 = R;
end

%% Output
model.iter = it;
model.gammas=gammas;
model.R=R;
model.X = X;
model.Z=Z;
end
