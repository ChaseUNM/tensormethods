using ITensors, ITensorMPS, BenchmarkTools, Random

include("hamiltonian(8-4).jl")
Random.seed!(42)

N = 8 
sites = siteinds("Qubit", N)
J = 1.0
g = 1.0
H_MPO = xxx_mpo(N, sites, J, g)
H_mat = xxx(N, J, g)
M = random_mps(sites;linkdims = [2, 4, 8, 16, 8, 4, 2])
M_vec = vectorize_mps(M; order = "reverse")
site = 1
M_block = M[site]*M[site + 1]

H_eff_2 = effective_Hamiltonian_2site(H_MPO, M, site)


@btime begin 
    H_contract = contract(H_eff_2)
end

@btime begin 
    H_contract2 = H_eff_2[end]
    for i in reverse(1:N - 1)
        H_contract2 = H_contract2*H_eff_2[i]
    end
end

@btime begin 
    ans = apply(H_MPO, M)
    for i in site + 2:N 
        ans[i] = ans[i]*conj(M[i])
    end 
    ans_contract = ans[end]
    for i in reverse(1:N-1)
        ans_contract = ans_contract*ans[i]
    end
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
    display(vec_itensor(ans_2; order = "reverse"))
    println("Difference: ", norm(ans_1 - ans_2))
    println("Test contraction: ")
    display(vec_itensor(ans_3; order = "reverse"))
    println("Difference 2: ", norm(ans_1 - ans_3))
end



