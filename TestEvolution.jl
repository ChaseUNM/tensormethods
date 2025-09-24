using LinearAlgebra, ITensors, ITensorMPS, Random 

Random.seed!(42)
include("BUG_tucker(8-27).jl")
include("Tucker_Matrices.jl")
include("Hamiltonian.jl")
include("BUG_small.jl")

N = 8
sites = siteinds("Qubit", N) 
g = 1.0
J = 1.0
H_mat = xxx(N, J, g)
# a = rand(2, 2)
# b = rand(2, 2)
# c = rand(2, 2)
# H_mat = kron(c, kron(b, a))
H_mat = xxx(N, J, g)
xxx_ops_list = xxx_ops(N, J, g)
total_H_xxx = total_H_itensor(xxx_ops_list, sites)
# abc_ops = ops_ex_3(a, b, c)
# abc_ops_H = total_H_itensor(abc_ops, sites)

# H_ten = ITensor(H_mat, sites, sites')


# H_mat = Matrix(1.0*I, 2^N, 2^N)
ident_ops = [identity_ops(N)]
total_H_ident = total_H_itensor(ident_ops, sites)
H_ten = ITensor(H_mat, sites, sites')

A = rand(ComplexF64, collect(fill(2, N))...)
core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)
core_arr, factors_arr = tucker(A; cutoff = 0.0)

alloc_tensors = preallocate_itensor(core_ten, factors_ten)

M_list, P_list, Y_list = pre_allocate(core_arr, factors_arr)

h = 0.01

@btime begin 
core_1,factors_1 = bug_step_itensor(H_ten, core_ten, factors_ten, h, sites)
end 

@btime begin 
core_2,factors_2 = bug_step_eff(total_H_xxx, core_ten, factors_ten, h, sites)
end 

@btime begin
core_3,factors_3 = bug_step_mat(xxx_ops_list, core_arr, factors_arr, h, M_list, P_list, Y_list)
end
ans_1 = reconstruct(core_1, factors_1)
ans_2 = reconstruct(core_2, factors_2)
ans_3 = Multi_TTM_recursive(core_3, factors_3)
println("Norm difference: ", norm(Array(ans_1, inds(ans_1)) - Array(ans_2, inds(ans_2))))
println("Norm difference 2: ", norm(Array(ans_1, inds(ans_1)) - ans_3))
