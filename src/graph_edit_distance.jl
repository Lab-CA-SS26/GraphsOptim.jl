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

function create_model_vars_reduced!(model::GenericModel, G::Graph, H::Graph, bidirectional::Bool = false)
    @variable(model, x[1:nv(G),1:nv(H)], Bin)

    # use a sparse indexed edge variable set. Edges are oriented to have only one edge per edge
    @variable(model,
              y[
                i in vertices(G), j in vertices(G), 
                k in vertices(H), l in vertices(H); 
                has_edge(G, i, j) && has_edge(H, k, l) &&
                i < j && (k < l || bidirectional)
              ],
              Bin
              )

    if bidirectional
        return OrientedVariables(x, y)
    else
        return ReducedVariables(x, y)
    end
end

create_model_vars_bidirectional!(model::GenericModel, G::Graph, H::Graph) = create_model_vars_reduced!(model, G, H, true)

function create_model_vars_full!(model::GenericModel, G::Graph, H::Graph)
    vars = create_model_vars_reduced!(model, G, H)
    @variable(model, nodeDelG[1:nv(G)], Bin)
    @variable(model, nodeDelH[1:nv(H)], Bin)

    @variable(model, edgeDelG[i in vertices(G), j in vertices(G); has_edge(G, i, j)], Bin)
    @variable(model, edgeDelH[k in vertices(H), l in vertices(H); has_edge(H, k, l)], Bin)

    return FullVariables(vars.x, vars.y, nodeDelG, nodeDelH, edgeDelG, edgeDelH)
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

function add_node_map_constraints!(model::GenericModel, vars::FullVariables, G, H)
    @constraint(model, [i in 1:nv(G)], sum(vars.x[i,:]) + vars.nodeDelG[i] == 1)
    @constraint(model, [j in 1:nv(H)], sum(vars.x[:,j]) + vars.nodeDelH[j] == 1)
end

function add_node_map_constraints!(model::GenericModel, vars::Union{ReducedVariables, OrientedVariables}, G, H)
    @constraint(model, [i in 1:nv(G)], sum(vars.x[i,:]) <= 1)
    @constraint(model, [j in 1:nv(H)], sum(vars.x[:,j]) <= 1)
end

function add_edge_map_constraints!(model::GenericModel, vars::FullVariables, G, H)
    @constraint(model, [i in vertices(G), j in vertices(G); has_edge(G, i, j) && i < j], 
                sum(vars.y[i, j, :, :]) + vars.edgeDelG[i, j] == 1)
    @constraint(model, [k in vertices(H), l in vertices(H); has_edge(H, k, l) && k < l], 
                sum(vars.y[:, :, k, l]) + vars.edgeDelH[k, l] == 1)
end
