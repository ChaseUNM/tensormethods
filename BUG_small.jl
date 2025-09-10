using LinearAlgebra, ITensors, ProgressMeter, BenchmarkTools

include("BUG_tucker(8-27).jl")
include("Hamiltonian.jl")
#Pratice BUG but don't form any large matrices
sz = [1 0; 0 -1]
sx = [0 1; 1 0]
sy = [0 -im; im 0]

function op_itensor(op, sites, ind)
    return ITensor(op, sites[ind], sites[ind]')
end

N = 3
A = rand(2^N)
sites = siteinds("Qubit", N)
cutoff = 1.0

core, factors = tucker_itensor(A, sites; cutoff = cutoff)

#Let H the xxx hamiltonian with J = -1, g = 1

core_inds = inds(core)
Q1, S1 = qr(core, core_inds[2], core_inds[3])
Q2, S2 = qr(core, core_inds[1], core_inds[3])
Q3, S3 = qr(core, core_inds[1], core_inds[2])

K1 = factors[1]*S1 
K2 = factors[2]*S2 
K3 = factors[3]*S3

function applyH_tucker(op_list, Q, factors, site)
    core_inds = inds(core)
    # not_site = setdiff(core_inds, [core_inds[site]])
    # Q, R = qr(core, not_site)
    new_factors = [] 
    N = length(factors)
    # println("N: $N")
    for i = 1:N
        # println("i: $i")
        if i == site
            # display(op_list[i])
            # display(factors[i])
            # println(op_list[i]*factors[i]) 
            push!(new_factors, op_list[i]*factors[i])
        else 
            # display(conj(factors[i]')*op_list[i]*factors[i])
            push!(new_factors, factors[i]*op_list[i]*conj(factors[i]'))
        end
        # println("New factors: ")
        # println(new_factors)
    end
    return new_factors 
end

function op_itensor_list(op_mat_list, sites) 
    N = length(op_mat_list)
    ops = Array{Any}(undef, N) 
    for i in 1:N 
        # op = op_itensor(op_mat_list, sites, i)
        op = ITensor(op_mat_list[i], sites[i], sites[i]')
        ops[i] = op 
    end
    return ops 
end

# K1_mat = rand(2,2)
# K1 = ITensor(K1_mat, inds(K1))

# Ident = Matrix(1.0*I, 2, 2)
# op = [sx, sz, sx]
# Q1_inds = inds(Q1)
# op_list = op_itensor_list(op, sites)

# factors_intermediate = [K1, factors[2], factors[3]]
# new_factors = applyH_tucker(op_list, Q1, factors_intermediate, 1)
# println("Answer: ")
# println(reconstruct(Q1, new_factors)*conj(Q1)')

# K1_mat = Array(K1, inds(K1))
# combine_Q = combiner(inds(Q1)[1], inds(Q1)[2])
# Q = Q1*combine_Q
# Q_mat = Array(Q, inds(Q))
# println("Answer 2")
# display(K1_mat*Q_mat*Q_mat') 

# A = zeros(2, 2, 2)
# A[:,:,1] = [1 2; 3 4]
# A[:,:,2] = [5 6; 7 8]
# A1 = ITensor(A, inds(Q1))
# A = A1*combine_Q 
# A_mat = Array(A, inds(A))
# # A_arr = reshape(Array(A1, inds(A1)), 2, 4)


# new_factors = applyH_tucker(op_list, A1, factors_intermediate, 1)
# println("Answer")
# ans1 = reconstruct(Q1, new_factors)*conj(Q1)'
# println(ans1)


# mat_factors = matricize_factors(factors)
# println("Answer 2")
# display(op[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*op[3]'*mat_factors[3], mat_factors[2]*op[2]'*mat_factors[2]))*Q_mat')

# #Now try summing 
# ops1 = [sx, Ident, Ident]
# ops2 = [Ident, sx, Ident]
# ops1_itensor = op_itensor_list(ops1, sites)
# ops2_itensor = op_itensor_list(ops2, sites)


# new_factors1 = applyH_tucker(ops1_itensor, Q1, factors_intermediate, 1)
# new_factors2 = applyH_tucker(ops2_itensor, Q1, factors_intermediate, 1)

# println("Answer: ")
# ans_sum = (reconstruct(Q1, new_factors1) + reconstruct(Q1, new_factors2))*conj(Q1)'
# println(ans_sum)
# println("Answer 2")
# display((ops1[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*ops1[3]'*mat_factors[3], mat_factors[2]*ops1[2]'*mat_factors[2])) + ops2[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*ops2[3]'*mat_factors[3], mat_factors[2]*ops2[2]'*mat_factors[2])))*Q_mat')

function xxx_ops(N,J,g)
    op_list = [] 
    #create zz chain
    for i in 1:N-1
        ops = [Ident for _ in 1:N]
        ops[i] = -J*sz 
        ops[i + 1] = sz
        push!(op_list, ops)
    end
    #create the x chain
    for i in 1:N 
        ops = [Ident for _ in 1:N]
        ops[i] = -g*J*sx
        push!(op_list, ops)
    end
    return op_list
end

function total_H_itensor(op_list_list, sites)
    N = length(op_list_list)
    op_list_list_itensor = Array{Any}(undef, N)
    for i in 1:N 
        op_list_list_itensor[i] = op_itensor_list(op_list_list[i], sites)
    end
    return op_list_list_itensor 
end

function K_evolution_mat(core, factors, site, K, ops_list)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K = factors[site]*S 
    K = Array(K, inds(K))
    C = combiner(not_site_inds;tags = "c")
    Q_arr = Q*C 
    Q_arr = Array(Q_arr, inds(Q_arr))

    N_ops = length(ops_list)
    mat_factors = matricize_factors(factors)
    init = ops_list[1][1]*K*Q_arr*kron(mat_factors[3]'*ops_list[1][3]'*mat_factors[3], mat_factors[2]*ops_list[1][2]'*mat_factors[2])
    for i in 2:N_ops
        init += ops_list[i][1]*K*Q_arr*kron(mat_factors[3]'*ops_list[i][3]'*mat_factors[3], mat_factors[2]*ops_list[i][2]'*mat_factors[2])
    end
    Kdot = init*Q_arr' 
    return Kdot
end
    

function K_evolution_itensor(core,factors,site, K, total_H,sites)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K_ten

    N_ops = length(total_H)
    new_factors = applyH_tucker(total_H[1], Q, factors_intermediate, site)
    # println("Q")
    # display(Q)
    # println("factors_intermediate")
    # display(factors_intermediate)
    init = reconstruct(Q, new_factors)
    # display(init)
    for i in 2:N_ops 
        new_factors = applyH_tucker(total_H[i], Q, factors_intermediate, site)
        init += reconstruct(Q, new_factors)
    end
    # display(init)
    K_dot = init*conj(Q')
    return K_dot
end

function C_dot_itensor(core, factors, total_H, sites)
    N_ops = length(total_H)
    N_factors = length(factors)
    mat_factors = matricize_factors(factors)
    desired_inds = prime(inds(core))
    init = ITensor(desired_inds)
    new_factors = []
    new_factors = Array{Any}(undef, N_factors)
    reconstruct_time = 0.0
    factor_multiple_time1 = 0.0
    factor_multiple_time2 = 0.0
    for i in 1:N_ops 
        for j in 1:N_factors 
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            mult_elapsed = @elapsed begin
                new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            end
            mult_elapsed2 = @elapsed begin 
                new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            end
            factor_multiple_time1 += mult_elapsed
            factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println(core)
        # println(new_factors)
        time_elapsed = @elapsed begin     
        init += reconstruct(core, new_factors)
        end
        reconstruct_time += time_elapsed
    end
    # display(new_factors)
    return init, reconstruct_time, factor_multiple_time1, factor_multiple_time2
end




# ops1 = [sx, Ident, Ident]
# ops2 = [Ident, sx, Ident]
# ops1_itensor = op_itensor_list(ops1, sites)
# ops2_itensor = op_itensor_list(ops2, sites)

# total_H = [ops1_itensor, ops2_itensor]
# ops_list = [ops1, ops2]

# Kdot = K_evolution_mat(core, factors, 1, K1_mat, ops_list)
# # Kdot_ten = K_evolution_itensor(core, factors, 1, K1_mat, total_H, sites)

# #Compare result to code in bug_step
xxx_ops_list = xxx_ops(N, 1, 1)
total_H_xxx = total_H_itensor(xxx_ops_list, sites)

H_mat = xxx(N, 1, 1)
H_ten = ITensor(H_mat, sites, sites')
# @btime begin
# core_f,factors_f = bug_step_itensor(H_ten, core, factors, 0.01, sites)
# end

# @btime begin 
# Kdot1 = K_evolution_itensor(core, factors, 1, K1, total_H_xxx, sites)
# Kdot2 = K_evolution_itensor(core, factors, 2, K2, total_H_xxx, sites)
# Kdot3 = K_evolution_itensor(core, factors, 3, K3, total_H_xxx, sites)
# end

#Now test the Cdot function 
#First create the "inefficient"? way to do it 
# @btime begin 
inter_C = H_ten*reconstruct(core,factors)
Cdot = reconstruct(inter_C, conj_factors2(factors))
# end
#Now test the "efficient"? way 
# @btime begin
@btime begin
Cdot_eff, reconstruct_time, mult_time, mult_time2 = C_dot_itensor(core, factors, total_H_xxx, sites)
println("reconstruct_time: ", reconstruct_time)
println("factor_multiple time: ", mult_time)
println("factor_multiple time2: ", mult_time2)
end
# println("Norm difference: ", norm(Cdot - Cdot_eff))