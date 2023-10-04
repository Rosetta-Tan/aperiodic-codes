using QuantumClifford

######################################################################
# Load code
######################################################################

filepath = "../data/toric_code/hz_d5.txt"
hz = []
open(filepath, "r") do file
    for line in eachline(file)
        # Split the line into binary elements using space as the delimiter
        elements = split(line)
        # Convert binary elements to integers (0 or 1)
        row = parse.(Int, elements)  #TODO: parse.
        # Append the row to the matrix
        push!(hz, row)  #TODO: push!
    end
end
hz = hcat(hz...)  #TODO: 1. hcat 2. ...

function read_binary_matrix(file_path::AbstractString)
    # Initialize an empty array to store the matrix
    matrix = []

    # Open the file for reading
    open(file_path, "r") do file
        for line in eachline(file)
            # Split the line into binary elements using space as the delimiter
            elements = split(line)
            
            # Convert binary elements to integers (0 or 1)
            row = parse.(Int, elements)
            
            # Append the row to the matrix
            push!(matrix, row)
        end
    end

    # Convert the array of arrays (matrix) into a Julia array
    matrix = hcat(matrix...)

    return matrix
end


