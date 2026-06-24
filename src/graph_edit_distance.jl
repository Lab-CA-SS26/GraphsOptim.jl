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

Variables = Union{ReducedVariables, FullVariables}

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

function add_F1_objective!(model, c::EditCosts, vars::FullVariables, G::Graph, H::Graph)
    @objective(model, Min, 
               sum(vars.x[i, k] * c.c_ik[i, k] for i in 1:nv(G) for k in 1:nv(H)) + 
               sum(vars.nodeDelG[i] * c.c_iε[i] for i in 1:nv(G)) + 
               sum(vars.nodeDelH[k] * c.c_εk[k] for k in 1:nv(H)) + 
               sum(vars.y[i, j, k, l] * c.c_ijkl[i, j, k, l] 
                   for i in 1:nv(G) for j in 1:nv(G)
                   for k in 1:nv(H) for l in 1:nv(H)
                   if has_edge(G, i, j) && has_edge(H, k, l) && i < j && k < l) +
               sum(vars.edgeDelG[i, j] * c.c_ijε[i, j]
                   for i in 1:nv(G) for j in 1:nv(G)
                   if has_edge(G, i, j) && i < j) +
               sum(vars.edgeDelH[k, l] * c.c_εkl[k, l]
                   for k in 1:nv(H) for l in 1:nv(H)
                   if has_edge(H, k, l) && k < l)
               )
end

function add_F2_objective!(model, c::EditCosts, vars::ReducedVariables, G::Graph, H::Graph, bidirectional::Bool = false)
    K = sum(c.c_iε[i] for i in 1:nv(G)) + 
        sum(c.c_εk[k] for k in 1:nv(H)) + 
        sum(c.c_ijε[i, j]
            for i in 1:nv(G) for j in 1:nv(G)
            if has_edge(G, i, j) && i < j) +
        sum(c.c_εkl[k, l]
            for k in 1:nv(H) for l in 1:nv(H)
            if has_edge(H, k, l) && k < l)
    @objective(model, Min, 
               sum(vars.x[i, k] * (c.c_ik[i, k] - c.c_iε[i] - c.c_εk[k]) for i in 1:nv(G) for k in 1:nv(H)) + 
               sum(vars.y[i, j, k, l] * (c.c_ijkl[i, j, k, l] - c.c_ijε[i, j] - c.c_εkl[k, l])
                   for i in 1:nv(G) for j in 1:nv(G)
                   for k in 1:nv(H) for l in 1:nv(H)
                   if has_edge(G, i, j) && has_edge(H, k, l) && i < j && (k < l || bidirectional)) +
               K
               )
end

add_FORI_objective!(model, c::EditCosts, vars::OrientedVariables, G::Graph, H::Graph) = add_F2_objective!(model, c, ReducedVariables(vars.x, vars.z), G, H, true)

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

function add_simple_topology_constraints!(model::GenericModel, vars::Variables, G, H)
    @constraint(model, [
                i in vertices(G), j in vertices(G), 
                k in vertices(H), l in vertices(H); 
                has_edge(G, i, j) && has_edge(H, k, l) &&
                i < j && k < l], 
                vars.y[i, j, k, l] <= vars.x[i, k] + vars.x[j, k])
    @constraint(model, [
                i in vertices(G), j in vertices(G), 
                k in vertices(H), l in vertices(H); 
                has_edge(G, i, j) && has_edge(H, k, l) &&
                i < j && k < l], 
                vars.y[i, j, k, l] <= vars.x[i, l] + vars.x[j, l])
end

function add_improved_topology_constraints_G_to_H!(model::GenericModel, vars::Variables, G, H)
    @constraint(model, [
                i in vertices(G), j in vertices(G), 
                k in vertices(H); 
                has_edge(G, i, j) && i < j], 
                sum(vars.y[i, j, k, :]) + sum(vars.y[i, j, :, k]) <= vars.x[i, k] + vars.x[j, k])
end

function add_improved_topology_constraints_H_to_G!(model::GenericModel, vars::Variables, G, H)
    @constraint(model, [
                k in vertices(H), l in vertices(H), 
                i in vertices(G); 
                has_edge(H, k, l) && k < l], 
                sum(vars.y[i, :, k, l]) + sum(vars.y[:, i, k, l]) <= vars.x[i, k] + vars.x[i, l])
end

function add_oriented_topology_constraints!(model::GenericModel, vars::OrientedVariables, G, H)
    # combines the constraints of add_improved_topology_constraints_G_to_H! and add_improved_topology_constraints_H_to_G!
    # but can be more precise due to the directionality of H
    @constraint(model, [
                k in vertices(H),
                i in vertices(G), j in vertices(G); 
                has_edge(G, i, j) && i < j], 
                sum(vars.z[i, j, k, :])  <= vars.x[i, k])
    @constraint(model, [
                k in vertices(H),
                i in vertices(G), j in vertices(G); 
                has_edge(G, i, j) && i < j], 
                sum(vars.z[i, j, :, k])  <= vars.x[j, k])
    @constraint(model, [
                k in vertices(H), l in vertices(H),
                i in vertices(G); 
                has_edge(H, k, l)], 
                sum(vars.z[i, :, k, l]) + sum(vars.z[:, i, l, k]) <= vars.x[i, k])
end

function construct_F1!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_full!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)
    add_edge_map_constraints!(model, vars, G, H)

    add_simple_topology_constraints!(model, vars, G, H)

    add_F1_objective!(model, c, vars, G, H)
    return model
end

function construct_F2minus!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_reduced!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)

    add_simple_topology_constraints!(model, vars, G, H)

    add_F2_objective!(model, c, vars, G, H)
end

function construct_F2!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_reduced!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)
    # note that edge map constraints are implied by the better topology constraints, so we can skip
    # them

    add_improved_topology_constraints_G_to_H!(model, vars, G, H)

    if ismissing(c)
        c = get_default_edit_costs(G, H)
    end

    add_F2_objective!(model, c, vars, G, H)
    return model
end

function construct_F2plus!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_reduced!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)
    # note that edge map constraints are implied by the better topology constraints, so we can skip
    # them

    add_improved_topology_constraints_G_to_H!(model, vars, G, H)
    add_improved_topology_constraints_H_to_G!(model, vars, G, H)

    add_F2_objective!(model, c, vars, G, H)
    return model
end

function construct_F1prime!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_full!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)
    # in the full variable set, we can't skip the edge map constraints
    add_edge_map_constraints!(model, vars, G, H)

    add_improved_topology_constraints_G_to_H!(model, vars, G, H)

    add_F1_objective!(model, c, vars, G, H)
    return model
end

function construct_F1plus!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_full!(model, G, H)

    add_node_map_constraints!(model, vars, G, H)
    # in the full variable set, we can't skip the edge map constraints
    add_edge_map_constraints!(model, vars, G, H)

    add_improved_topology_constraints_G_to_H!(model, vars, G, H)
    add_improved_topology_constraints_H_to_G!(model, vars, G, H)

    add_F1_objective!(model, c, vars, G, H)
    return model
end

function construct_FORI!(model, G, H, c::EditCosts = get_default_edit_costs(G, H))
    vars = create_model_vars_bidirectional!(model, G, H)
    add_node_map_constraints!(model, vars, G, H)

    add_oriented_topology_constraints!(model, vars, G, H)

    add_FORI_objective!(model, c, vars, G, H)
    return model
end
