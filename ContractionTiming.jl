using ITensors, ITensorMPS, BenchmarkTools, Random

include("hamiltonian(8-4).jl")
include("tdvp(8-4).jl")
Random.seed!(42)

N = 4 
sites = siteinds("Qubit", N)
J = 1.0
g = 1.0
H_MPO = xxx_mpo(N, sites, J, g)
H_mat = xxx(N, J, g)
M = random_mps(sites;linkdims = [2, 4, 2])
M_vec = vectorize_mps(M; order = "reverse")
site = 1
M_block = M[site]*M[site + 1]

H_eff_2 = effective_Hamiltonian_2site(H_MPO, M, site)


# @btime begin 
#     H_contract = contract(H_eff_2)
# end

# @btime begin 
#     H_contract2 = H_eff_2[end]
#     for i in reverse(1:N - 1)
#         H_contract2 = H_contract2*H_eff_2[i]
#     end
# end

# @btime begin 
#     ans = apply(H_MPO, M)
#     for i in site + 2:N 
#         ans[i] = ans[i]*conj(M[i])
#     end 
#     ans_contract = ans[end]
#     for i in reverse(1:N-1)
#         ans_contract = ans_contract*ans[i]
#     end
# end
function fast_contraction(H)
    N = length(H)
    H_contract = H[end]
    for i in reverse(1: N - 1)
        H_contract = H_contract*H[i]
    end
    return H_contract 
end

let 
    H_contract = contract(H_eff_2)
    H_contract2 = H_eff_2[end]
    for i in reverse(1:N - 1)
        H_contract2 = H_contract2*H_eff_2[i]
    end
    ans_1 = noprime(H_contract*M_block) 
    ans_2 = noprime(H_contract2*M_block)
    ans_3 = apply(H_MPO, M)
    for i in site + 2:N 
        ans_3[i] = ans_3[i]*conj(M[i])
    end
    ans_3 = contract(ans_3)
    println("Inefficient contraction: ")
    display(vec_itensor(ans_1; order = "reverse"))
    println("More efficient contraction: ")
    display(vec_itensor(ans_2; order = "natural"))
    println("Difference: ", norm(ans_1 - ans_2))
    println("Test contraction: ")
    display(vec_itensor(ans_3; order = "reverse"))
    println("Difference 2: ", norm(ans_1 - ans_3))
end

# H_eff_mat, M_mat = conversion(H_eff_2, M_block)

# println("Mat_vec contraction: ")
# display(H_eff_mat*M_mat)


# H_contract = contract(H_eff_2)
# H_contract_fast = fast_contraction(H_eff_2)

# H_mat, M_vec = conversion_short(H_contract, M_block)
# H_mat_fast, M_vec_fast = conversion_short(H_contract_fast, M_block)

# ans = exp(-im*H_mat)*M_vec 
# ans_fast = exp(-im*H_mat_fast)*M_vec_fast 

# comm_inds = commoninds(H_contract, M_block)
# comm_inds_fast = commoninds(H_contract_fast, M_block)

# ans_ten = ITensor(ans, comm_inds)
# ans_fast_ten = ITensor(ans_fast, comm_inds_fast)

#pratice with combiner 
# comm_inds = commoninds(H_contract2, M_block)
# notcomm_inds = uniqueinds(H_contract2, M_block)
# col_ind = combiner(comm_inds; tags="columns")
# row_ind = combiner(notcomm_inds; tags = "rows")
# H_combine = H_contract*col_ind*row_ind
# M_combine = M_block*col_ind
# ans = H_combine*M_combine
# row_inds, col_inds = inds(H_combine)
# H_mat = Array(H_combine, (row_inds, col_inds))
# println("onto M_vec")
# M_vec = Array(M_combine, row_inds)
# println("New contraction: ")
# display(H_mat*M_vec)
# println("Old Contraction: ")
# display(vec_itensor(H_contract2*M_block; order = "reverse"))
# println("norm difference: ",  norm(H_mat*M_vec - vec_itensor(H_contract2*M_block; order = "reverse")))

#Get left indices and right indices
function lr_inds(M, site_number)
    l_inds = Index{Int64}[] 
    r_inds = Index{Int64}[]
    N = length(inds(M))
    for i in 1:N
        if hastags(inds(M)[i], "n = $site_number") == true || hastags(inds(M)[i], "l = $(site_number - 1)") == true
            push!(l_inds, inds(M)[i])
        else 
            push!(r_inds, inds(M)[i])
        end
    end
    return l_inds, r_inds 
end


