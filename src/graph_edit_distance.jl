struct ReducedVariables
    x::Matrix{VariableRef}
    y::SparseAxisArray{VariableRef}
end

struct FullVariables
    x::Matrix{VariableRef}
    y::SparseAxisArray{VariableRef}
    nodeDelG::Vector{VariableRef}
    nodeDelH::Vector{VariableRef}
    edgeDelG::SparseAxisArray{VariableRef}
    edgeDelH::SparseAxisArray{VariableRef}
end

struct OrientedVariables
    x::Matrix{VariableRef}
    z::SparseAxisArray{VariableRef}
end

# based on FORI implementation of the cost function
struct EditCosts
    c_ik::Array{Number, 2}
    c_iε::Vector{Number}
    c_εk::Vector{Number}
    # note that for sparse graphs we might want to use a sparse array here
    c_ijkl::Array{Number, 4}
    c_ijε::Array{Number, 2}
    c_εkl::Array{Number, 2}
end

# asserts that the given cost function could belong to G and H in terms of their dimensions
# This function simultaneously serves as a documentation on EditCosts
function validate_cost_function(c::EditCosts, G::Graph, H::Graph)
    @assert size(c.c_ik) == (nv(G), nv(H))
    @assert size(c.c_iε) == (nv(G),)
    @assert size(c.c_εk) == (nv(H),)
    @assert size(c.c_ijkl) == (nv(G), nv(G), nv(H), nv(H))
    @assert size(c.c_ijε) == (nv(G), nv(G))
    @assert size(c.c_εkl) == (nv(H), nv(H))
end

# returns the intuitive "deleting things costs 1" cost function
function get_default_edit_costs(G::Graph, H::Graph)
    return EditCosts(
        zeros(Int, nv(G), nv(H)),
        ones(Int, nv(G)),
        ones(Int, nv(H)),
        zeros(Int, nv(G), nv(G), nv(H), nv(H)),
        ones(Int, nv(G), nv(G)),
        ones(Int, nv(H), nv(H))
    )
end
