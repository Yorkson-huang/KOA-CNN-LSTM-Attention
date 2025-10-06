% Kepler Optimization Algorithm (KOA)
function [Sun_Score, Best_Pos, KOA_curve, bestPred, bestNet, bestInfo] = KOA(SearchAgents_no, Tmax, ub, lb, dim)

%% Initialization
Sun_Pos   = zeros(1, dim);  %% Best position found so far (the "sun")
Sun_Score = inf;            %% Best objective value found so far

%% Control parameters
Tc = 3;
M0 = 0.1;
lambda = 15;

%% Step 1: Randomly initialize the population
orbital   = rand(1, SearchAgents_no);                %% Orbital eccentricity (Eq. 4)
T         = abs(randn(1, SearchAgents_no));          %% Orbital period (Eq. 5)
Positions = initialization(SearchAgents_no, dim, ub, lb);
t = 0;  %% Function evaluation counter

%% Evaluate initial population
for i = 1:SearchAgents_no
    [PL_Fit(i), tsmvalue{i}, tnet{i}, tinfo{i}] = objectiveFunction(Positions(i,:)');
    if PL_Fit(i) < Sun_Score
        Sun_Score = PL_Fit(i);
        Sun_Pos   = Positions(i,:);
        bestPred  = tsmvalue{i};
        bestNet   = tnet{i};
        bestInfo  = tinfo{i};
    end
end

while t < Tmax
    Order        = sort(PL_Fit);
    worstFitness = Order(SearchAgents_no);           %% Eq. 11
    M            = M0 * exp(-lambda * (t / Tmax));   %% Eq. 12

    %% Compute Euclidean distance between each planet and the sun
    for i = 1:SearchAgents_no
        R(i) = 0;
        for j = 1:dim
            R(i) = R(i) + (Sun_Pos(j) - Positions(i,j))^2;  %% Eq. 7
        end
        R(i) = sqrt(R(i));
    end

    %% Compute mass values (Eq. 8 and 9)
    for i = 1:SearchAgents_no
        denom = 0;
        for k = 1:SearchAgents_no
            denom = denom + (PL_Fit(k) - worstFitness);
        end
        MS(i) = rand * (Sun_Score - worstFitness) / denom;
        m(i)  = (PL_Fit(i) - worstFitness) / denom;
    end

    %% Step 2: Gravitational force (Eq. 6, 24)
    Rnorm  = (R - min(R)) / (max(R) - min(R) + eps);
    MSnorm = (MS - min(MS)) / (max(MS) - min(MS) + eps);
    Mnorm  = (m - min(m)) / (max(m) - min(m) + eps);
    for i = 1:SearchAgents_no
        Fg(i) = orbital(i) * M * ((MSnorm(i) * Mnorm(i)) / (Rnorm(i)^2 + eps)) + rand;
    end

    %% Semi-major axis for each orbit (Eq. 23)
    for i = 1:SearchAgents_no
        a1(i) = rand * (T(i)^2 * (M * (MS(i) + m(i)) / (4 * pi * pi)))^(1/3);
    end

    for i = 1:SearchAgents_no
        a2 = -1 - (rem(t, Tmax / Tc) / (Tmax / Tc));    %% Eq. 29
        n  = (a2 - 1) * rand + 1;                        %% Eq. 28
        a  = randi(SearchAgents_no);
        b  = randi(SearchAgents_no);
        rd = rand(1, dim);
        r  = rand;

        U1 = rd < r;              %% Eq. 21
        previousPos = Positions(i,:);

        if rand < rand
            %% Step 6: update distance using adaptive factor (Eq. 26-27)
            h  = 1 / exp(n * randn);
            Xm = (Positions(b,:) + Sun_Pos + Positions(i,:)) / 3.0;
            Positions(i,:) = Positions(i,:) .* U1 + (Xm + h .* (Xm - Positions(a,:))) .* (1 - U1);
        else
            %% Step 3: velocity update (Eq. 13-22)
            if rand < 0.5
                f = 1;
            else
                f = -1;
            end
            L = (M * (MS(i) + m(i)) * abs((2 / (R(i) + eps)) - (1 / (a1(i) + eps))))^0.5; %% Eq. 15
            U = rd > rand(1, dim);
            if Rnorm(i) < 0.5
                Mtemp = rand .* (1 - r) + r;                 %% Eq. 16
                l  = L * Mtemp .* U;                         %% Eq. 14
                Mv = rand .* (1 - rd) + rd;                  %% Eq. 20
                l1 = L .* Mv .* (1 - U);                     %% Eq. 19
                V(i,:) = l .* (2 * rand * Positions(i,:) - Positions(a,:)) ...
                    + l1 .* (Positions(b,:) - Positions(a,:)) ...
                    + (1 - Rnorm(i)) * f * U1 .* rand(1, dim) .* (ub - lb);   %% Eq. 13a
            else
                U2 = rand > rand;                             %% Eq. 22
                V(i,:) = rand .* L .* (Positions(a,:) - Positions(i,:)) ...
                    + (1 - Rnorm(i)) * f * U2 * rand(1, dim) .* (rand * ub - lb); %% Eq. 13b
            end

            if rand < 0.5
                f = 1;
            else
                f = -1;
            end

            %% Step 5: position update (Eq. 25)
            Positions(i,:) = (Positions(i,:) + V(i,:) .* f) ...
                + (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i,:));
        end

        %% Boundary control
        if rand < rand
            for j = 1:dim
                if Positions(i,j) > ub(j) || Positions(i,j) < lb(j)
                    Positions(i,j) = lb(j) + rand * (ub(j) - lb(j));
                end
            end
        else
            Positions(i,:) = min(max(Positions(i,:), lb), ub);
        end

        %% Evaluate new position
        [PL_Fit1, tsmvalue1, tnet1, tinfo1] = objectiveFunction(Positions(i,:)');

        %% Step 7: elitism (Eq. 30)
        if PL_Fit1 < PL_Fit(i)
            PL_Fit(i)   = PL_Fit1;
            tsmvalue{i} = tsmvalue1;
            tnet{i}     = tnet1;
            tinfo{i}    = tinfo1;

            if PL_Fit(i) < Sun_Score
                Sun_Score = PL_Fit(i);
                Sun_Pos   = Positions(i,:);
                bestPred  = tsmvalue{i};
                bestNet   = tnet{i};
                bestInfo  = tinfo{i};
            end
        else
            Positions(i,:) = previousPos;
        end

        t = t + 1;
        if t > Tmax
            break;
        end
        t = t + 1;
        if t > Tmax
            break;
        end
    end

    if t > Tmax
        break;
    end
end

KOA_curve = sortrows(PL_Fit', -1);  %% Sorted fitness values (descending)

%% Discretize best position (keep learning rate continuous)
for i = 1:size(Sun_Pos,2)
    if i == 1
        Best_Pos(i) = Sun_Pos(i);
    else
        Best_Pos(i) = round(Sun_Pos(i));
    end
end
end

%% Helper: random initialization inside bounds
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end
