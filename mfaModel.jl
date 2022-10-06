using Turing
using Random


# Define MFA model. (port from Jupyter notebook)

"""
Define custom type for Dirichlet Allocation
"""
mutable struct DirichletAllocation
    name
    outputs
    nParams
    DirichletAllocation(name, outputs) = new(name, outputs, length(outputs))
end

mutable struct SinkProcess
    name
    outputs
    nParams
    SinkProcess(name) = new(name, [], 0)
end

"""
@Name(x)
Returns the value of the variable itself and a string corresponding to the variable name. 
For full details, refer this post: https://discourse.julialang.org/t/convert-input-function-variable-name-to-string/25398
"""
macro Name(x)
    quote
    ($(esc(x)), $(string(x)))
    end
end



"""
Set Dirichlet Allocation Process Priors for all processes.
"""

allProcessDescription = Dict(
:Import_Iron_Ore        => DirichletAllocation("Import_Iron_Ore_Allocation",       ["Iron_Ore_Consumption"])                                  
, :Iron_Ore_Production    => DirichletAllocation("Iron_Ore_Production_Allocation",   ["Export", "Iron_Ore_Consumption"])    
, :Iron_Ore_Consumption   => DirichletAllocation("Iron_Ore_Consumption_Allocation",  ["Blast_Furnace","DRI_Production","Other"])
, :Blast_Furnace          => DirichletAllocation("Blast_Furnace_Allocation",         ["Loss","Pig_Iron"])
, :Import_Scrap           => DirichletAllocation("Import_Scrap_Allocation",          ["Scrap_Consumption"])
, :Purchased_Scrap        => DirichletAllocation("Purchased_Scrap_Allocation",       ["Scrap_Collected"])
, :Scrap_Collected        => DirichletAllocation("Scrap_Collected_Allocation",       ["Export","Scrap_Consumption"])
, :Scrap_Consumption      => DirichletAllocation("Scrap_Consumption_Allocation",     ["Blast_Furnace", "Basic_Oxygen_Furnace","Electric_Arc_Furnace","Cupola", "Other_Casting"])
, :Import_Pig_Iron        => DirichletAllocation("Import_Pig_Iron_Allocation",       ["Pig_Iron_Consumption"])
, :Pig_Iron               => DirichletAllocation("Pig_Iron_Allocation",              ["Export","Pig_Iron_Consumption"]) 
, :Pig_Iron_Consumption   => DirichletAllocation("Pig_Iron_Consumption_Allocation",  ["Basic_Oxygen_Furnace","Electric_Arc_Furnace","Cupola", "Other_Casting"]) 
, :Import_DRI             => DirichletAllocation("Import_DRI_Allocation",            ["DRI_Consumption"])        
, :DRI_Production         => DirichletAllocation("DRI_Production_Allocation",        ["Loss", "DRI"])
, :DRI                    => DirichletAllocation("DRI_Allocation",                   ["Export", "DRI_Consumption"])
, :DRI_Consumption        => DirichletAllocation("DRI_Consumption_Allocation",       ["Blast_Furnace", "Basic_Oxygen_Furnace","Electric_Arc_Furnace", "Cupola", "Other_Casting"])
, :Basic_Oxygen_Furnace   => DirichletAllocation("Basic_Oxygen_Furnace_Allocation",  ["Continuous_Casting","Loss"])  
, :Electric_Arc_Furnace   => DirichletAllocation("Electric_Arc_Furnace_Allocation",  ["EAF_Yield","Loss"]) 
, :EAF_Yield              => DirichletAllocation("EAF_Yield_Allocation",             ["Continuous_Casting", "Ingot_Casting"])
, :Cupola                 => DirichletAllocation("Cupola_Allocation",                ["Ingot_Casting", "Loss"])
, :Continuous_Casting     => DirichletAllocation("Continuous_Casting_Allocation",    ["CC_Yield", "CC_Loss"]) 
, :CC_Loss                => DirichletAllocation("CC_Loss_Allocation",               ["CC_Yield", "Loss"]   )
, :CC_Yield               => DirichletAllocation("CC_Yield_Allocation",              ["Hot_Strip_Mill", "Plate_Mill", "Rod_and_Bar_Mill", "Section_Mill"]) 
, :Ingot_Casting          => DirichletAllocation("Ingot_Casting_Allocation",         ["IC_Yield", "IC_Loss"]) 
, :IC_Loss                => DirichletAllocation("IC_Loss_Allocation",               ["IC_Yield", "Loss"]   )
, :IC_Yield               => DirichletAllocation("IC_Yield_Allocation",              ["Primary_Mill"]) 
, :Other_Casting          => DirichletAllocation("Other_Casting_Allocation",         ["OC_Yield", "OC_Loss"]) 
, :OC_Loss                => DirichletAllocation("OC_Loss_Allocation",               ["OC_Yield", "Loss"]   )
, :OC_Yield               => DirichletAllocation("IC_Yield_Allocation",              ["Steel_Product_Casting","Iron_Product_Casting"])   
, :Hot_Strip_Mill         => DirichletAllocation("Hot_Strip_Mill_Allocation",     ["HSM_Yield", "Scrap_Consumption"])     
, :HSM_Yield              => DirichletAllocation("HSM_Yield_Allocation",          ["Cold_Rolling_Mill", "Hot_Rolled_Sheet","Pipe_Welding_Plant"]) 
, :Cold_Rolling_Mill      => DirichletAllocation("Cold_Rolling_Mill_Allocation",  ["CRM_Yield", "Scrap_Consumption"])
, :CRM_Yield              => DirichletAllocation("CRM_Yield_Allocation",          ["Cold_Rolled_Sheet", "Galvanized_Plant", "Tin_Mill"])
, :Rod_and_Bar_Mill       => DirichletAllocation("Rod_and_Bar_Mill_Allocation",   ["RBM_Yield", "Scrap_Consumption"]) 
, :RBM_Yield              => DirichletAllocation("RBM_Yield_Allocation",          ["Seamless_Tube_Plant", "Bars", "Reinforcing_Bars", "Wire_and_Wire_Rods", "Light_Section"]) 
, :Section_Mill           => DirichletAllocation("Section_Mill_Allocation",       ["SM_Yield", "Scrap_Consumption"]) 
, :SM_Yield               => DirichletAllocation("SM_Yield_Allocation",           ["Rails_and_Rail_Accessories", "Heavy_Section"]) 
, :Primary_Mill           => DirichletAllocation("Primary_Mill_Allocation",       ["PM_Yield", "Scrap_Consumption"])
, :PM_Yield               => DirichletAllocation("PM_Yield_Allocation",           ["Export", "Hot_Strip_Mill", "Cold_Rolling_Mill", "Plate_Mill", "Rod_and_Bar_Mill","Section_Mill"])
, :Plate_Mill             => DirichletAllocation("Plate_Mill_Allocation",         ["Plates", "Scrap_Consumption"]) 
, :Tin_Mill               => DirichletAllocation("Tin_Mill_Allocation",           ["Tin_Mill_Products", "Scrap_Consumption"])      
, :Galvanized_Plant       => DirichletAllocation("Gal_Plant_Allocation",          ["Galvanized_Sheet", "Scrap_Consumption"])     
, :Pipe_Welding_Plant     => DirichletAllocation("Pipe_Welding_Plant_Allocation", ["Pipe_and_Tubing",  "Scrap_Consumption"])
, :Seamless_Tube_Plant    => DirichletAllocation("Seamless_Tube_Allocation",      ["Pipe_and_Tubing",  "Scrap_Consumption"])
, :Cold_Rolled_Sheet      => DirichletAllocation("Cold_Rolled_Sheet_Allocation",   ["Export", "Automotive",   "Machinery", "Steel_Products", "Scrap_Consumption"])
, :Galvanized_Sheet       => DirichletAllocation("Galvanized_Sheet_Allocation",    ["Export", "Construction", "Automotive", "Scrap_Consumption"])
, :Tin_Mill_Products      => DirichletAllocation("Tin_Mill_Products_Allocation",   ["Export", "Automotive",   "Steel_Products", "Scrap_Consumption"])
, :Hot_Rolled_Sheet       => DirichletAllocation("Hot_Rolled_Sheet_Allocation",    ["Export", "Construction", "Automotive", "Machinery", "Energy", "Steel_Products", "Scrap_Consumption"])                                                                                    
, :Pipe_and_Tubing        => DirichletAllocation("Pipe_and_Tubing_Allocation",     ["Export", "Construction", "Automotive", "Machinery", "Energy",   "Scrap_Consumption"])                                                              
, :Plates                 => DirichletAllocation("Plates_Allocation",              ["Export", "Construction", "Automotive", "Machinery", "Energy",   "Scrap_Consumption"])
, :Reinforcing_Bars       => DirichletAllocation("Reinforcing_Bars_Allocation",    ["Export", "Construction", "Scrap_Consumption"])
, :Bars                   => DirichletAllocation("Bars_Allocation",                ["Export", "Construction", "Automotive", "Machinery", "Energy",   "Scrap_Consumption"])
, :Wire_and_Wire_Rods     => DirichletAllocation("Wires_and_Wire_Rods_Allocation", ["Export", "Construction", "Automotive", "Machinery", "Energy",   "Scrap_Consumption"])
, :Rails_and_Rail_Accessories => DirichletAllocation("Rails_and_Rail_Accessories_Allocation",  ["Export", "Construction", "Machinery", "Scrap_Consumption"])
, :Light_Section              => DirichletAllocation("Light_Sections_Allocation",              ["Export", "Construction", "Automotive", "Scrap_Consumption"])  
, :Heavy_Section              => DirichletAllocation("Heavy_Sections_Allocation",              ["Export", "Construction", "Scrap_Consumption"])    
, :Steel_Product_Casting      => DirichletAllocation("Steel_Casting_Allocation",               [ "Construction", "Automotive","Machinery", "Energy"])
, :Iron_Product_Casting       => DirichletAllocation("Iron_Casting_Allocation",                [ "Construction", "Automotive","Machinery", "Energy"])
, :Ingot_Import               => DirichletAllocation("Import_of_Ingot",  ["Primary_Mill"])  
, :Intermediate_Product_Import => DirichletAllocation("Intermediate_Product_Import_Allocation", ["Cold_Rolled_Sheet", "Galvanized_Sheet", "Tin_Mill_Products", "Hot_Rolled_Sheet", "Pipe_and_Tubing", "Plates", "Reinforcing_Bars","Bars","Wire_and_Wire_Rods", "Rails_and_Rail_Accessories", "Light_Section","Heavy_Section", "Steel_Product_Casting", "Iron_Product_Casting"])
, :Export         => SinkProcess("Sink1")
, :Loss           => SinkProcess("Sink2")
, :Other          => SinkProcess("Sink3")
, :Construction   => SinkProcess("Sink4")
, :Automotive     => SinkProcess("Sink5")
, :Machinery      => SinkProcess("Sink6")
, :Energy         => SinkProcess("Sink7")
, :Steel_Products => SinkProcess("Sink8")
)

# nm = names(Main)
# allProcessesFinal = setdiff(nm, [:Base, :Core, :InteractiveUtils, :Main, :ans, :nm, :DirichletAllocation, :SinkProcess])
# allProcessesString = string.(allProcessesFinal)
# println(allProcessesFinal)

input_observations = (
    Import_DRI = 2.47,
    Import_Iron_Ore =             5.16,
    Import_Pig_Iron =             4.27,
    Import_Scrap =                3.72,
    Ingot_Import =                6.94,
    Intermediate_Product_Import = 23.46,
    Iron_Ore_Production =          54.7,
    Purchased_Scrap =             70.98,
    )

inputDefs = Dict(
    "Import_DRI" => 2.47,
    "Import_Iron_Ore" =>             5.16,
    "Import_Pig_Iron" =>             4.27,
    "Import_Scrap" =>                3.72,
    "Ingot_Import" =>                6.94,
    "Intermediate_Product_Import" => 23.46,
    "Iron_Ore_Production" =>          54.7,
    "Purchased_Scrap" =>             70.98,
    )



function dirPrior(shares, concentration=nothing; with_stddev=nothing)
    if (!isnothing(concentration) && !isnothing(with_stddev)) || (isnothing(concentration) && isnothing(with_stddev))
        error("Specify one of concentration or stddev")

    end

    factor = sum(shares)
    shares = shares ./ factor

    if !isnothing(with_stddev)
        i, stddev = with_stddev
        stddev /= factor
        mi = shares[i + 1]
        limit = sqrt(mi * (1 - mi) / (1 + length(shares)))
        concentration = (mi * (1 - mi) / stddev^2) - 1
        if !isfinite(concentration)
            concentration = 1e10
        end
        
    else
        concentration = length(shares) * concentration
    end

    return concentration * shares
end

paramDefs = Dict(
    "Iron_Ore_Production"=> dirPrior(([1.39, 6.82])/sum(([1.39, 6.82]))*100; with_stddev = [0, 12]),
    "Iron_Ore_Consumption"=> dirPrior(([17.25, 1.73, 0.60])/sum(([17.25, 1.73, 0.60]))*100; with_stddev = [0, 7]),

    "DRI_Production"      => dirPrior(([0.41, 1.73])/sum(([0.41, 1.73]))*100; with_stddev = [0, 22]),
    "DRI"                 => dirPrior(([0.17, 0.73])/sum(([0.17, 0.73]))*100; with_stddev = [0, 28] ),
    "DRI_Consumption"     => dirPrior(([1.6, 1.09, 21.47, 0.1, 0.77])/sum(([1.6, 1.09, 21.47, 0.1, 0.77]))*100; with_stddev=[0, 5]),
    "Scrap_Collected"     => dirPrior(([1.16, 4.18])/sum(([1.16, 4.18]))*100; with_stddev = [0, 16]),
    "Scrap_Consumption"   => dirPrior(([ 0.90, 1.62, 5.82, 0.59, 0.1 ])/sum(([0.90, 1.62, 5.82, 0.59, 0.1 ]))*100; with_stddev=[0, 9]), 
    "Blast_Furnace"       => dirPrior(([2.42, 2.13])/sum(([2.42, 2.13]))*100; with_stddev = [0, 21]),
    "Pig_Iron"            => dirPrior(([0.79, 6.66])/sum(([0.79, 6.66]))*100; with_stddev = [0, 11]),
    "Pig_Iron_Consumption"=> dirPrior(([0.94, 0.27, 0.24, 0.13])/sum(([0.94, 0.27, 0.24, 0.13]))*100; with_stddev = [0, 28]),
    "Basic_Oxygen_Furnace"=> dirPrior(([22.20,  3.44])/sum(([22.20,  3.44])) * 100; with_stddev = [0, 6]),
    "Electric_Arc_Furnace"=> dirPrior(([22.20,  3.44])/sum(([22.20,  3.44])) * 100; with_stddev = [0, 6]),
    "Continuous_Casting"  => dirPrior(([30, 1.55])/sum(([30, 1.55])) * 100; with_stddev = [0, 4]),
    "Ingot_Casting"       => dirPrior(([30, 1.55])/sum(([30, 1.55])) * 100; with_stddev = [0, 4]),
    "Other_Casting"       => dirPrior(([30, 1.55])/sum(([30, 1.55])) * 100; with_stddev = [0, 4]),
    "CC_Loss"             => dirPrior(([30,  4.43])/sum(([30,  4.43])) * 100; with_stddev = [0, 6]),
    "IC_Loss"             => dirPrior(([30,  4.43])/sum(([30,  4.43])) * 100; with_stddev = [0, 6]),
    "OC_Loss"             => dirPrior(([30,  4.43])/sum(([30,  4.43])) * 100; with_stddev = [0, 6]),
    "CC_Yield"            => dirPrior(([11.46, 2.11, 2.82, 1.81])/sum(([11.46, 2.11, 2.82, 1.81])) * 100; with_stddev = [0, 11]),
    "Plate_Mill"          => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Tin_Mill"            => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Hot_Strip_Mill"      => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Cold_Rolling_Mill"   => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Rod_and_Bar_Mill"    => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Section_Mill"        => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Primary_Mill"        => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Galvanized_Plant"    => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Pipe_Welding_Plant"  => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
    "Seamless_Tube_Plant" => dirPrior(([30,  2.73])/sum(([30,  2.73]))* 100; with_stddev = [0, 5]),
)

            
observationsRatio = [
    (["Cold_Rolled_Sheet"], ["Automotive"],        0.250),  
    (["Cold_Rolled_Sheet"], ["Machinery"],         0.079),  
    (["Cold_Rolled_Sheet"], ["Steel_Products"],    0.313), 
    (["Cold_Rolled_Sheet"], ["Export"],            0.112), 
    (["Galvanized_Sheet"],  ["Construction"],      0.19),  
    (["Galvanized_Sheet"],  ["Automotive"],        0.42), 
    (["Galvanized_Sheet"],  ["Export"],            0.15),  
    (["Hot_Rolled_Sheet"],  ["Construction"],      0.59 ),  
    (["Hot_Rolled_Sheet"],  ["Automotive"],        0.133 ), 
    (["Hot_Rolled_Sheet"],  ["Machinery"],         0.108 ),  
    (["Hot_Rolled_Sheet"],  ["Energy"],            0.01  ), 
    (["Hot_Rolled_Sheet"],  ["Steel_Products"],    0.0027), 
    (["Hot_Rolled_Sheet"],  ["Export"],            0.065),
    (["Pipe_and_Tubing"],   ["Construction"],       0.227), 
    (["Pipe_and_Tubing"],   ["Automotive"],         0.08), 
    (["Pipe_and_Tubing"],   ["Machinery"],          0.04),  
    (["Pipe_and_Tubing"],   ["Energy"],             0.55), 
    (["Pipe_and_Tubing"],   ["Export"],             0.065),
    (["Plates"],            ["Construction"],       0.0408), 
    (["Plates"],            ["Automotive"],         0.01), 
    (["Plates"],            ["Machinery"],          0.5187),  
    (["Plates"],            ["Energy"],             0.067), 
    (["Plates"],            ["Export"],             0.231),
    (["Bars"],              ["Construction"],       0.152), 
    (["Bars"],              ["Automotive"],         0.311), 
    (["Bars"],              ["Machinery"],          0.238),  
    (["Bars"],              ["Energy"],             0.046), 
    (["Bars"],              ["Export"],             0.131),
    (["Reinforcing_Bars"],  ["Construction"],       0.925),
    (["Reinforcing_Bars"],  ["Export"],             0.039),
    (["Tin_Mill_Products"], ["Automotive"],         0.006),
    (["Tin_Mill_Products"], ["Steel_Products"],     0.685),
    (["Tin_Mill_Products"], ["Export"],             0.067),
    (["Wire_and_Wire_Rods"],  ["Construction"],     0.388), 
    (["Wire_and_Wire_Rods"],  ["Automotive"],       0.285), 
    (["Wire_and_Wire_Rods"],  ["Machinery"],        0.1),  
    (["Wire_and_Wire_Rods"],  ["Energy"],           0.049), 
    (["Wire_and_Wire_Rods"],  ["Export"],           0.094),
    (["Rails_and_Rail_Accessories"],  ["Construction"],  0.779), 
    (["Rails_and_Rail_Accessories"],  ["Machinery"],     0.047), 
    (["Rails_and_Rail_Accessories"],  ["Export"],        0.141),
    (["Light_Section"],          ["Construction"],   0.86),
    (["Light_Section"],          ["Automotive"],     0.026),
    (["Light_Section"],          ["Export"],         0.057),
    (["Heavy_Section"],          ["Construction"],   0.877),
    (["Heavy_Section"],          ["Export"],         0.092),
    (["Steel_Product_Casting"],  ["Construction"],   0.259), 
    (["Steel_Product_Casting"],  ["Automotive"],     0.385), 
    (["Steel_Product_Casting"],  ["Machinery"],      0.259),
    (["Iron_Product_Casting"],   ["Construction"],   0.311), 
    (["Iron_Product_Casting"],   ["Automotive"],     0.552), 
    (["Iron_Product_Casting"],   ["Machinery"],      0.066),
]


observations = [
    (["Iron_Ore_Production"],       ["Export"],               11.2 ),  
    (["Iron_Ore_Consumption"],      ["Blast_Furnace"],        46.3), 
    (["Blast_Furnace"],             ["Pig_Iron"],             32.1), 
    (["DRI"],                       ["Export"],               0.01),  
    (["DRI_Consumption"],           ["Blast_Furnace"],        0.049), 
    (["DRI_Consumption"],           ["Basic_Oxygen_Furnace"], 1.91), 
    (["DRI_Consumption"],           ["Electric_Arc_Furnace"], 1.62), 
    (["DRI_Consumption"],           ["Cupola"],               0.01 ), 
    (["DRI_Consumption"],           ["Other_Casting"],        0.01 ),
    (["Pig_Iron"],                  ["Export"],               0.021),  
    (["Pig_Iron_Consumption"],      ["Basic_Oxygen_Furnace"], 31.5), 
    (["Pig_Iron_Consumption"],      ["Electric_Arc_Furnace"], 5.79), 
    (["Pig_Iron_Consumption"],      ["Cupola"],               0.057), 
    (["Pig_Iron_Consumption"],      ["Other_Casting"],        0.046),
    (["Scrap_Collected"],           ["Export"],               21.4 ), 
    (["Scrap_Consumption"],         ["Blast_Furnace"],        2.62 ),          
    (["Scrap_Consumption"],         ["Basic_Oxygen_Furnace"], 8.35 ) ,  
    (["Scrap_Consumption"],         ["Electric_Arc_Furnace"], 50.9) ,      
    (["Scrap_Consumption"],         ["Cupola"],               1.11 ),                    
    (["Scrap_Consumption"],         ["Other_Casting"],        0.167),
    (["Basic_Oxygen_Furnace"],      ["Continuous_Casting"],   36.281),        
    (["Electric_Arc_Furnace"],      ["EAF_Yield"],            52.414),        
    (["Pipe_Welding_Plant"],        ["Pipe_and_Tubing"],      2.165),                   
    (["Seamless_Tube_Plant"],       ["Pipe_and_Tubing"],      2.162),
    (["HSM_Yield"],                 ["Hot_Rolled_Sheet"],      19.544),    
    (["CRM_Yield"],                 ["Cold_Rolled_Sheet"],     11.079),    
    (["Tin_Mill"],                  ["Tin_Mill_Products"],     2.009),
    (["Galvanized_Plant"],          ["Galvanized_Sheet"],      16.749),
    (["Plate_Mill"],                ["Plates"],                 9.12),
    (["RBM_Yield"],                 ["Reinforcing_Bars"],           5.65),
    (["RBM_Yield"],                 ["Bars"],                       6.7),
    (["RBM_Yield"],                 ["Wire_and_Wire_Rods"],         2.784),
    (["RBM_Yield"],                 ["Light_Section"],              2.13 ),   
    (["SM_Yield"],                  ["Heavy_Section"],              5.03),   
    (["SM_Yield"],                  ["Rails_and_Rail_Accessories"], 1.009),
]


# Elicitation from a single expert
input_mu    = [3.18,  4.15,   3.69, 3.41,  9.1,    25,   47.87,  71.74]
input_sigma = [1.48,  1.27 ,  1.7,  1.75,  3.94,   10,   9.69,   7.49]
input_lower = [0,     2,      2.5,  2,     0,      10,   20,     50 ]
input_upper = [5,     7,      7,    6,     20,     45,   70,     90]    

struct Determin{T<:Real} <: ContinuousUnivariateDistribution
  val::T
end

struct DeterminVec{T<:Vector} <: MultivariateDistribution
    val::T
end

Distributions.rand(rng::AbstractRNG, d::Determin) = d.val
Distributions.logpdf(d::Determin, x::T) where T<:Real = zero(x)

Distributions.rand(rng::AbstractRNG, d::DeterminVec) = d.val
Distributions.logpdf(d::DeterminVec, x::T) where T<:Vector = zero(x)

"""
Define transfer functions for Dirichlet and Sink Process.
"""
transferFunctions(process::DirichletAllocation, params) = params
transferFunctions(process::SinkProcess, params) = nothing



"""
Build matrices from process params and input distributions
"""
function buildMatrices(processes, processParams, inputs, possible_inputs; ::Type{TC} = Array{Float64, 2}) where {TC}
    Np = length(processParams)

    # transferCoeffs = TC(undef, Np, Np)
    transferCoeffs = zeros(TC, Np, Np)

    pids = Dict(sort(string.(keys(processes))) .=> collect(1:length(keys(processes))))

    for (pid, process) in processes
        if !(haskey(processParams, pid))
            continue
        params = processParams[pid]
        process_tcs = transferFunctions(process, params)
        if !(isempty(process.outputs))
            dest_idx = [pids[dest_id] for dest_id in process.outputs]
            transferCoeffs[dest_idx, pids[pid]] ~ process_tcs
        end
    
    end

    possible_inputs_idx = [pids[k] for k in possible_inputs]
    allInputs = zeros(Np)
    allInputs[possible_inputs_idx] = inputs
    return transferCoeffs, allInputs
end

"""
Define a function that takes in observations, prior related data etc and builds our probabilistic model Turing Style!!
"""

@model function MFA(processes, input_defs, param_defs, inputμ, inputσ, inputLB, inputUB; flow_observations=nothing, input_observations=nothing, ratio_observations=nothing, inflow_observations = nothing)

    possible_inputs = sort(string.(keys(input_defs)))

    if !isnothing(flow_observations)
        σ ~ truncated(Normal(0, 0.15), 0, 0.5)
    end

    if !isnothing(input_observations)
        σ_input ~ truncated(Normal(0, 0.15), 0, 0.5)
    end

    if !isnothing(ratio_observations)
        σ_ratio ~ truncated(Normal(0, 0.15), 0, 0.5)
    end


    # we also want to create distributions for each of the process IDs, using their original names as variable names!
    
    processParams = Dict()
    for k in keys(processes)
        if string(k) in keys(param_defs)
            defs = param_defs[string(k)]
        else
            defs = ones(processes[k].nparams)
        end
        @assert length(defs) = processes[k].nparams
        if processes[k] isa DirichletAllocation
            if length(defs) > 1
                # @eval $k ~ Dirichlet(defs)
                # processParams[k] = @eval $k ~ Dirichlet(defs)
                processParams[k] ~ Dirichlet(defs)
            else
                # @eval $k ~ Determin(1)
                # processParams[k] = @eval $k ~ Determin(1)
                processParams[k] ~ Determin(1)
            end

        elseif processes[k] isa SinkProcess
            @eval $k ~ nothing
            # processParams[k] = @eval $k ~ nothing 
            processParams[k] = nothing
        end
    end

    
    inputs = Vector{UnivariateDistribution}(undef, length(input_defs))
    for i in 1:length(input_defs)
        inputs[i] ~ truncated(Normal(inputμ[i], inputσ[i]), inputLB[i], inputUB[i])
    end

    transferCoeffs, allInputs = buildMatrices(processes, processParams, inputs, possible_inputs)
    
    # Convert to distributions
    m, n = size(transferCoeffs)
    transferCoeffsFinal = zeros(m, n)
    for mIdx in 1:m
        for nIdx in 1:n
            transferCoeffsFinal[mIdx, nIdx] ~ DeterminVec(transferCoeffs[mIdx, nIdx])
        end
    end

    process_throughputs ~ DeterminVec((I - transferCoeffsFinal) \ allInputs)

    flows ~ DeterminVec(transferCoeffsFinal' * process_throughputs)

    # add in observations and return final posterior variables.

end

