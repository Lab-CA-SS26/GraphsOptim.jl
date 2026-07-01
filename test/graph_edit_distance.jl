using GraphsOptim, Graphs, JuMP, HiGHS
using Test

all_formulations = [F1, F1prime, F1plus, F2, F2minus, F2plus, FORI]

@testset "Small Graph" begin
    G = Graph(3)
    add_edge!(G, 1, 2)
    
    H = Graph(3)
    add_edge!(H, 2, 3)
    add_edge!(H, 1, 3)

    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 1
    end
end

@testset "Cost Functions Simple" begin
    G = Graph(3)
    add_edge!(G, 1, 2)
    
    H = Graph(3)
    add_edge!(H, 2, 3)
    add_edge!(H, 1, 3)
    c_origin = GraphsOptim.get_default_edit_costs(G, H)
    
    # increating costs for node subsitutions 
    c = GraphsOptim.EditCosts(
        ones(Int, nv(G), nv(H)),
        5 * c_origin.c_iε,
        5 * c_origin.c_εk,
        c_origin.c_ijkl,
        c_origin.c_ijε,
        c_origin.c_εkl
    )
    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; c=c, formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 4
    end
    # increating costs for edge subsitutions 
    c = GraphsOptim.EditCosts(
        c_origin.c_ik,
        c_origin.c_iε,
        c_origin.c_εk,
        c_origin.c_ijkl,
        100 * c_origin.c_ijε,
        10 * c_origin.c_εkl
    )
    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; c=c, formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 10
    end

    # force a specific suboptimal node map with edit costs 
    G = Graph(3)
    add_edge!(G, 2, 3)
    
    H = Graph(4)
    add_edge!(H, 1, 3)
    add_edge!(H, 3, 4)
    c_origin = GraphsOptim.get_default_edit_costs(G, H)
    custom_substitution_cost = 100 .+ c_origin.c_ik
    custom_substitution_cost[1,1] = 0
    custom_substitution_cost[2,2] = 0
    custom_substitution_cost[3,3] = 0
    c = GraphsOptim.EditCosts(
        custom_substitution_cost,
        c_origin.c_iε,
        c_origin.c_εk,
        c_origin.c_ijkl,
        c_origin.c_ijε,
        c_origin.c_εkl
    )
    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        vars = GraphsOptim.edit_distance!(model, G, H; c=c, formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 4
        @test value(model[:x][1,1]) == 1
        @test value(model[:x][2,2]) == 1
        @test value(model[:x][3,3]) == 1
    end
end

@testset "Edgecases" begin
    G = Graph(3)
    
    H = Graph(3)
    add_edge!(H, 1, 2)

    # one graph empty
    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 1

        # reverse order
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, H, G; formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 1
    end

    # both graphs empty
    H = Graph(5)
    for formulation in all_formulations
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 2
    end
    
    # no nodes in one graph
    H = Graph(0)
    for formulation in [F1, F1prime, F1plus]
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; formulation=formulation)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 3
    end
    # for F2 type formulations, the formulation contains no variables
    # we thus add a dummy variable to make the solver accept the model
    for formulation in [F2, F2minus, F2plus, FORI]
        model = Model(HiGHS.Optimizer)
        GraphsOptim.edit_distance!(model, G, H; formulation=formulation)
        @test num_variables(model) == 0
        @variable(model, dummy)
        set_silent(model)
        optimize!(model)
        @test objective_value(model) == 3
    end
end
