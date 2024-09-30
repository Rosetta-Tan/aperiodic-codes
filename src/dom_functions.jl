# -----
# Types 
# -----

(@isdefined LatType) || (@enum LatType mem_mem mmm_mmm eee_eee Tennis1 Tennis2 HalfHalf1 HalfHalf2 Manual OneM OneE Triangular Bulk mem_eee eee_mem mee_eme)
(@isdefined BoundaryType) || (@enum BoundaryType mType eType None)
(@isdefined LeftRight) || (@enum LeftRight Left Right)

Cart = Tuple{Int64, Int64, Int64}
Vacancy = Tuple{Cart, Cart, BoundaryType}

# --------------
# Main Functions 
# --------------
"""
    specifyStabilisers(dim::Tuple{Int64, Int64, Int64}; type::LatType=mem_mem, 
    manualX::NTuple{6, Int64}, NTuple{6, Int64})

Given a prism dimension, and a particular cube *type*, output a list
of indices that specify the position of X and Z stabilisers, which can
be passed to `computeStabilisers()`.

`manualX` etc. is a 6-tuple of either 1 (X plaquettes) or 0 (no X plaquettes) on
the boundaries (x, y, z, -x, -y, -z)
"""
function specifyStabilisers(dim::Tuple{Int64, Int64, Int64}; type::LatType=mem_mem, 
            manualX::NTuple{6, Int64}=(0,0,0,0,0,0), manualZ::NTuple{6, Int64}=(0,0,0,0,0,0))
  stabX = Int[];
  stabZ = Int[]; 
  dimx, dimy, dimz = dim;
  sdimx, sdimy, sdimz = dim .+ 1; # If there are 3 qubits, there are 4 stabilisers
  # @assert all(dim .> 1) "Must be at least 2³ qubits"

  if type == mem_mem
    # X stabilisers 
    # - Bottom corners and edges
    push!(stabX, 1, 2, (sdimy-1)*sdimx+1, (sdimy-1)*sdimx+2, sdimy*sdimx+1,
          2*sdimy*sdimx-sdimx+1);
    # - Bulk and x faces and negative z face
    indices = (x for x in sdimx:(sdimx*(sdimz-1)*sdimy)
               if (m = mod(x, sdimx*sdimy);(m > sdimx) & (m <= sdimx*(sdimy-1))));
    push!(stabX, indices...);
    # - Top z face
    indices = (sdimy*sdimx*(sdimz-1)+sdimx+1):(sdimy*sdimx*sdimz-sdimx-1);
    push!(stabX, indices...);

    # Z stabilisers 
    indices = (x for x in (sdimx*sdimy+1):(sdimx*sdimy*(sdimz-1)) if (mod(x, sdimx) != 0) &
               (mod(x, sdimx) != 1));
    push!(stabZ, indices...);
    push!(stabZ, (sdimx*sdimz*sdimy) .- [0, 1, sdimx, sdimx*sdimy]...);
  elseif type == mmm_mmm 
    stabX = collect(1:(sdimx*sdimz*sdimy));
    indices = map(stabX) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz); 
      (1 < x < sdimx) & (1 < y < sdimy) & (1 < z < sdimz)
    end;
    stabZ = stabX[indices]; 
  elseif type == eee_eee 
    stabZ = collect(1:(sdimx*sdimz*sdimy));
    indices = map(stabZ) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz); 
      (1 < x < sdimx) & (1 < y < sdimy) & (1 < z < sdimz)
    end;
    stabX = stabZ[indices]; 
  elseif type == Bulk
    stabZ = collect(1:(sdimx*sdimz*sdimy));
    stabX = collect(1:(sdimx*sdimz*sdimy));
  elseif type == Tennis2 
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x < sdimx) & (1 <= y <= sdimy) & (1 <= z < sdimz)
    end 
    push!(stabX, xyzToLoc.([(1,1,1), (sdimx,1,1), (1,2,1), (1,1,2), (sdimx,2,1), (sdimx,1,2)], 
                           sdimx, sdimy, sdimz)...);

    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 <= x <= sdimx) & (1 < y < sdimy) & (1 < z <= sdimz)
    end 
    push!(stabZ, xyzToLoc.([(sdimx,1,sdimz), (sdimx,sdimy,sdimz-1), (sdimx-1,sdimy,sdimz), 
                            (sdimx,sdimy,sdimz), (sdimx,1,sdimz-1), (sdimx-1,1,sdimz)], 
                           sdimx, sdimy, sdimz)...);
  elseif type == Tennis1 
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 <= x <= sdimx) & (1 < y < sdimy) & (1 < z <= sdimz) & ((x,y,z) != (sdimx, sdimy-1, sdimz))
    end
    push!(stabX, xyzToLoc.([(1,1,1), (2,1,1), (1,2,1), (1,1,2)], sdimx, sdimy, sdimz)...);
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x < sdimx) & (1 <= y <= sdimy) & (1 <= z < sdimz) & ((x,y,z) != (2,1,1))
    end 
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,sdimz-1), (sdimx,sdimy-1,sdimz), 
                            (sdimx-1,sdimy,sdimz),(sdimx,sdimy,sdimz)], sdimx, sdimy, sdimz)...);
  elseif type == HalfHalf1 
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 <= x < sdimx) & (1 <= y < sdimy) & (1 <= z < sdimz)
    end;
    push!(stabX, xyzToLoc.([(sdimx,1,1), (sdimx,2,1), (sdimx,1,2),
                            (1,sdimy,1), (2,sdimy,1), (1,sdimy,2),
                            (1,1,sdimz), (2,1,sdimz), (1,2,sdimz)], 
                           sdimx, sdimy, sdimz)...);
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x <= sdimx) & (1 < y <= sdimy) & (1 < z <= sdimz)
    end;
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,1), (sdimx-1,sdimy,1), (sdimx,sdimy-1,1),
                            (1,sdimy,sdimz), (1,sdimy-1,sdimz), (1,sdimy,sdimz-1),
                            (sdimx,1,sdimz), (sdimx-1,1,sdimz), (sdimx,1,sdimz-1)],
                           sdimx, sdimy, sdimz)...);
  elseif type == HalfHalf2 
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x <= sdimx) & (1 < y <= sdimy) & (1 < z <= sdimz)
    end;
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 <= x < sdimx) & (1 <= y < sdimy) & (1 <= z < sdimz)
    end;
  elseif type == Manual 
    full = 1:(sdimx*sdimz*sdimy);
    XLower = 2 .- manualX[4:end];
    XUpper = (sdimx, sdimy, sdimz) .+ manualX[1:3] .- 1;
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      all(XLower .<= (x,y,z) .<= XUpper)
    end; 
    ZLower = 2 .- manualZ[4:end];
    ZUpper = (sdimx, sdimy, sdimz) .+ manualZ[1:3] .- 1;
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      all(ZLower .<= (x,y,z) .<= ZUpper)
    end; 
  elseif type == OneE
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (y < sdimy) && !(y == (sdimy-1) && x == sdimx && z == sdimz)
    end;
    push!(stabX, xyzToLoc.([(1,sdimy,1), (2,sdimy,1), (1,sdimy,2)], 
                           sdimx, sdimy, sdimz)...);
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x < sdimx) && (1 < y <= sdimy) && (1 < z < sdimz) 
    end; 
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,sdimz), (sdimx,sdimy,sdimz-1),
                            (sdimx-1,sdimy,sdimz)], 
                          sdimx, sdimy, sdimz)...);
  elseif type == OneM 
    full = 1:(sdimx*sdimz*sdimy);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 < x < sdimx) && (1 < y <= sdimy) && (1 < z < sdimz)
    end;
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      y < sdimy
    end; 
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,sdimz), (sdimx,sdimy,sdimz-1),
                            (sdimx-1,sdimy,sdimz)],
                           sdimx, sdimy, sdimz)...);
  elseif type == Triangular 
    @assert sdimx == sdimy == sdimz
    sd = sdimx; 
    full = 1:(sdimx*sdimy*sdimz);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      ((1 < x < sdimx) && (1 < y < sdimy) && (1 < z < sdimz)) || 
      ((x == 1) && (y+z <= sd+1)) || 
      ((y == 1) && (x+z <= sd+1)) || 
      ((z == 1) && (x+y <= sd+1)) || 
      ((x == sd) && (1 < y < sd-1) && (1 < z <= sd-y)) || 
      ((y == sd) && (1 < x < sd-1) && (1 < z <= sd-x)) || 
      ((z == sd) && (1 < x < sd-1) && (1 < y <= sd-x))
    end; 
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      ((1 < x < sdimx) && (1 < y < sdimy) && (1 < z < sdimz)) || 
      ((x == 1) && (2 < y < sd) && (sd-y+2 <= z < sd)) || 
      ((y == 1) && (2 < x < sd) && (sd-x+2 <= z < sd)) || 
      ((z == 1) && (2 < x < sd) && (sd-x+2 <= y < sd)) || 
      ((x == sd) && (sd+1 <= y+z)) || 
      ((y == sd) && (sd+1 <= x+z)) || 
      ((z == sd) && (sd+1 <= x+y)) 
    end;
  elseif type == mem_eee
    full = 1:(sdimx*sdimy*sdimz);
    stabX = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      ((1 < x <= sdimx) && (1 < y < sdimy) && (1 < z < sdimz)) || 
        ((1 < x < sdimx) && (1 < y < sdimy) && (z == sdimz)) || 
        ((x == sdimx) && (1 < y < sdimy-1) && (z == sdimz))
    end;
    stabZ = filter(full) do loc 
      x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
      (1 <= x < sdimx) && (1 <= y <= sdimy) && (1 <= z < sdimz)
    end; 
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,sdimz), (sdimx,sdimy,sdimz-1),
                            (sdimx-1,sdimy,sdimz), (sdimx,sdimy-1,sdimz)],
                            sdimx, sdimy, sdimz)...);
  elseif type == eee_mem
    full = 1:(sdimx*sdimy*sdimz);
    stabX = filter(full) do loc 
        x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
        (1 <= x < sdimx) && (1 <= y < sdimy) && (1 < z < sdimz)
    end;
    push!(stabX, xyzToLoc.([(1,1,1), (2,1,1), (1,2,1), 
                            (1,1,sdimz), (2,1,sdimz), (1,2,sdimz)],
                            sdimx, sdimy, sdimz)...);
    stabZ = filter(full) do loc 
        x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
        (1 < x <= sdimx) && (1 < y <= sdimy) && (1 <= z <= sdimz)
    end;
    push!(stabZ, xyzToLoc.([(sdimx,1,sdimz),(sdimx,1,sdimz-1),(sdimx-1,1,sdimz),
                            (1,sdimy,sdimz),(1,sdimy,sdimz-1),(1,sdimy-1,sdimz)],
                            sdimx, sdimy, sdimz)...);
  elseif type == mee_eme
    full = 1:(sdimx*sdimy*sdimz);
    stabX = filter(full) do loc 
        x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
        (1 < x <= sdimx) && (1 <= y < sdimy) && (1 < z < sdimz)
    end;
    push!(stabX, xyzToLoc.([(1,1,1), (2,1,1), (1,2,1), (1,1,2)],
                            sdimx, sdimy, sdimz)...);
    stabZ = filter(full) do loc 
        x,y,z = locToXYZ(loc, sdimx, sdimy, sdimz);
        !((x,y,z) == (1,2,1)) && 
            (1 <= x < sdimx) && (1 < y <= sdimy) && (1 <= z <= sdimz)
    end;
    push!(stabZ, xyzToLoc.([(sdimx,sdimy,sdimz),(sdimx,sdimy,sdimz-1),(sdimx,sdimy-1,sdimz)],
                            sdimx, sdimy, sdimz)...);
  else 
      error("Unrecognsied type, $type.")
  end 
  return sort(stabX), sort(stabZ)
end
"""
    specifyStabilisers(dim::Int64; type::LatType=mem_mem)
"""
function specifyStabilisers(dim::Int64; type::LatType=mem_mem)
  specifyStabilisers((dim, dim, dim); type=type)
end

"""
    computeStabilisers(dim::Cart, stX::Vector{Int64}, stZ::Vector{Int64};
    pbcs::Tuple{Bool,Bool,Bool} = (false,false,false))

Given a specification of stabilisers, computes the `S` matrix. 
"""
function computeStabilisers(dim::Cart, stX::Vector{Int64}, stZ::Vector{Int64};
        pbcs::Tuple{Bool,Bool,Bool}=(false,false,false), 
        vacancies::Vector{Vacancy}=Vacancy[])
  npoints = prod(dim) |> Int;
  sdim = dim .+ 1;
  sdimx, sdimy, sdimz = sdim;
    
  full = 1:prod(sdim);
  Xs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (loc ∈ stX || (all(1 .< (x,y,z) .< sdim))) && 
        all(map(enumerate(pbcs)) do (index, pbc)
                !pbc || (x,y,z)[index] < sdim[index]
        end) && 
        all(map(vacancies) do (pos, size, btype) 
          (btype == mType) || !all((pos) .<= (x,y,z) .<= (pos .+ size))
        end)
  end;
  Zs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (loc ∈ stZ || (all(1 .< (x,y,z) .< sdim))) && 
        all(map(enumerate(pbcs)) do (index, pbc)
                !pbc || (x,y,z)[index] < sdim[index]
            end) && 
        all(map(vacancies) do (pos, size, btype) 
          (btype == eType) || !all((pos) .<= (x,y,z) .<= (pos .+ size))
            end)
  end;

  # Pad with an extra qubit on each face to construct the stabilisers on
  # the boundaries
  lx, ly, lz = dim .+ 2;
  kX, kZ = length(Xs), length(Zs);

  # Matrices to store the acted on states
  ρ0 = zeros(Int, lx, ly, lz); 

  # S matrix to store the stabilisers 
  S = zeros(Int, 4*npoints, kX+kZ);

  # Create X stabilisers
  for (index, loc) in enumerate(Xs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    σ_indices = [
               (0,0,0),
               (1,1,0),
               (1,0,1),
               (0,1,1)];
    μ_indices = [
               (0,0,0),
               (1,0,0),
               (0,1,0),
               (0,0,1)];
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    σ_indices = map(σ_indices) do (x,y,z)
        if pbcs[1]
            if x == 1
                x = sdimx;
            elseif x >= lx
                x = x - lx + 2;
            end
        end
        if pbcs[2]
            if y == 1
                y = sdimy;
            elseif y >= ly
                y = y - ly + 2;
            end
        end
        if pbcs[3]
            if z == 1
                z = sdimz;
            elseif z >= lz
                z = z - lz + 2;
            end
        end
        (x,y,z)
    end;
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    μ_indices = map(μ_indices) do (x,y,z) 
        if pbcs[1]
            if x == 1
                x = sdimx;
            elseif x >= lx
                x = x - lx + 2;
            end
        end
        if pbcs[2]
            if y == 1
                y = sdimy;
            elseif y >= ly
                y = y - ly + 2;
            end
        end
        if pbcs[3]
            if z == 1
                z = sdimz;
            elseif z >= lz
                z = z - lz + 2;
            end
        end
        (x,y,z)
    end;
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to S
    S[1:2*npoints, index] = vcat(S_σ, S_μ); 
  end

  # Z stabilisers
  for (index, loc) in enumerate(Zs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    σ_indices = [
               (1,1,0),
               (1,0,1),
               (1,1,1),
               (0,1,1)];
    μ_indices = [
               (1,0,0),
               (0,1,0),
               (0,0,1),
               (1,1,1)];
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    σ_indices = map(σ_indices) do (x,y,z) 
        if pbcs[1]
            if x == 1
                x = sdimx;
            elseif x >= lx
                x = x - lx + 2;
            end
        end
        if pbcs[2]
            if y == 1
                y = sdimy;
            elseif y >= ly
                y = y - ly + 2;
            end
        end
        if pbcs[3]
            if z == 1
                z = sdimz;
            elseif z >= lz
                z = z - lz + 2;
            end
        end
        (x,y,z)
    end;
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    μ_indices = map(μ_indices) do (x,y,z) 
        if pbcs[1]
            if x == 1
                x = sdimx;
            elseif x >= lx
                x = x - lx + 2;
            end
        end
        if pbcs[2]
            if y == 1
                y = sdimy;
            elseif y >= ly
                y = y - ly + 2;
            end
        end
        if pbcs[3]
            if z == 1
                z = sdimz;
            elseif z >= lz
                z = z - lz + 2;
            end
        end
        (x,y,z)
    end;
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to S 
    S[2*npoints+1:end, index + kX] = vcat(S_σ, S_μ); 
  end

  if isempty(vacancies)
    return MatrixSpace(GF(2), 4*(npoints), (kX+kZ))(S);
  end
                                                                
  full = 1:npoints;
  vacancyqubits = filter(full) do loc
    x,y,z = locToXYZ(loc, dim...);
    any(map(vacancies) do (pos, size, btype)
      all(pos .<= (x,y,z) .< (pos .+ size))
    end)
  end;
  vacancysubset = qubitsToSubset(vacancyqubits, npoints);

  return MatrixSpace(GF(2), 4*(npoints - length(vacancyqubits)), (kX+kZ))(
    S[filter(1:end) do t t ∉ vacancysubset end,:]
  );
end;

"""
    computeEdgeDislocStabilisers(
      dim::Tuple{Int64,Int64,Int64},
      defects::Vector{Tuple{Int64, BoundaryType, BoundaryType}};
      boundary::Vector{BoundaryType}=BoundaryType[],
      pad::Int64=1
    )

Computes stabilisers for an edge dislocation in the z direction, with periodic 
boundary conditions in the x direction. The dislocation is such that the bottom
twist has the 3-side on the positive y side. 

Example: 
```
computeEdgeDislocStabilisers((2,2,4), 
  defects=Tuple{Int64,BoundaryType,BoundaryType}[
    (2, mType, eType)
  ],
  boundary=BoundaryType[mType, eType],
  pad=1
)
```
"""
function computeEdgeDislocStabilisers(dim::Tuple{Int64,Int64,Int64}, 
    defects::Vector{Tuple{Int64, BoundaryType, BoundaryType}};
    boundary::Vector{BoundaryType}=BoundaryType[],
    pad::Int64=1)
  npoints = prod(dim) |> Int;
  sdim = dim .+ 1;
  sdimx, sdimy, sdimz = sdim;
  poss = getindex.(defects, 1);

  full = 1:prod(sdim);
  Xs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (x < sdimx) &&
      (mType ∈ boundary || ((1<y<sdimy) && (1<z<sdimz))) && 
      all(map(defects) do (pos, t1, t2)
            (y != pos) || 
                (
                    (z != sdimz-pad) && 
                    ((t1 == mType) || (z != pad+1)) && 
                    ((t2 == mType) || (z != sdimz-pad-1))
                )
          end)
  end;
  Zs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (x < sdimx) && 
      (eType ∈ boundary || ((1<y<sdimy) && (1<z<sdimz))) && 
      all(map(defects) do (pos, t1, t2)
            (y != pos) || 
                (
                    (z != sdimz-pad) && 
                    ((t1 == eType) || (z != pad+1)) && 
                    ((t2 == eType) || (z != sdimz-pad-1))
                )
          end)
  end;

  # Pad with an extra qubit on each face to construct the stabilisers on
  # the boundaries
  lx, ly, lz = dim .+ 2;
  kX, kZ = length(Xs), length(Zs);

  # Matrices to store the acted on states
  ρ0 = zeros(Int, lx, ly, lz); 

  # S matrix to store the stabilisers 
  S = zeros(Int, 4*npoints, kX+kZ);

  # Create X stabilisers
  for (index, loc) in enumerate(Xs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    if (y ∉ poss) || (z <= pad) || (z >= (sdimz-pad+1)) 
      # Regular bulk
      σ_indices = [
                   (0,0,0),
                   (1,1,0),
                   (1,0,1),
                   (0,1,1)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (0,1,0),
                   (0,0,1)];
    elseif any(map(defects) do (pos, t1, t2) 
                 (t1 == mType) && (y == pos) && (z == (pad+1))
               end)
      # Bottom of the twist
      σ_indices = [
                   (0,0,0),
                   (1,1,0),
                   (1,0,1),
                   (1,1,1),
                   (0,1,1),
                   (0,1,2)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (0,1,0),
                   (0,0,1),
                   (0,1,1)];
    elseif any(map(defects) do (pos, t1, t2)
                 (t2 == mType) && (y == pos) && (z == (sdimz-pad-1))
               end)
      # Top of the twist
      σ_indices = [
                   (0,0,0),
                   (0,0,1),
                   (1,0,1),
                   (1,1,1),
                   (1,0,2),
                   (0,1,2)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (1,0,1),
                   (0,1,1),
                   (0,0,2)];
    elseif (y ∈ poss)
      # In the twist
      σ_indices = [
                   (0,0,0),
                   (1,1,1),
                   (1,0,1),
                   (0,1,2)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (0,1,1),
                   (0,0,1)];
    else 
      @error "Shouldn't be here X"
    end
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;
    
    if x == 1
      σ_indices = [(sdim[1]-1,0,0) .+ ind for ind in σ_indices];
      μ_indices = [(sdim[1]-1,0,0) .+ ind for ind in μ_indices];
      ρ_σ[CartesianIndex.(σ_indices)] .= 1;
      ρ_μ[CartesianIndex.(μ_indices)] .= 1;
    end

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to S 
    S[1:2*npoints, index] = vcat(S_σ, S_μ); 
  end

  # Z stabilisers
  for (index, loc) in enumerate(Zs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    if (y ∉ poss) || (z <= pad) || (z >= (sdimz-pad+1)) 
      σ_indices = [
                   (1,1,0),
                   (1,0,1),
                   (1,1,1),
                   (0,1,1)];
      μ_indices = [
                   (1,0,0),
                   (0,1,0),
                   (0,0,1),
                   (1,1,1)];
    elseif any(map(defects) do (pos, t1, t2) 
                 (t1 == eType) && (y == pos) && (z == (pad+1))
               end)
            # Bottom
      σ_indices = [
                   (1,1,0),
                   (1,0,1),
                   (0,1,1),
                   (1,1,2),
                   (0,1,2)];
      μ_indices = [
                   (1,0,0),
                   (0,1,0),
                   (0,0,1),
                   (0,1,1),
                   (1,1,1),
                   (1,1,2)];
    elseif any(map(defects) do (pos, t1, t2)
                 (t2 == eType) && (y == pos) && (z == (sdimz-pad-1))
               end)
            # Top
      σ_indices = [
                   (1,0,1),
                   (1,1,1),
                   (1,0,2),
                   (0,1,2),
                   (1,1,2)];
      μ_indices = [
                   (1,0,0),
                   (0,0,1),
                   (1,0,1),
                   (0,1,1),
                   (0,0,2),
                   (1,1,2)];
    elseif y ∈ poss
      σ_indices = [
                   (1,1,1),
                   (1,0,1),
                   (1,1,2),
                   (0,1,2)];
      μ_indices = [
                   (1,0,0),
                   (0,0,1),
                   (0,1,1),
                   (1,1,2)];
    else 
      @error "Shouldn't be here Z"
    end
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;
    
    if x == 1
      σ_indices = [(sdim[1]-1,0,0) .+ ind for ind in σ_indices];
      μ_indices = [(sdim[1]-1,0,0) .+ ind for ind in μ_indices];
      ρ_σ[CartesianIndex.(σ_indices)] .= 1;
      ρ_μ[CartesianIndex.(μ_indices)] .= 1;
    end

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to S 
    S[2*npoints+1:end, index + kX] = vcat(S_σ, S_μ); 
  end

  return MatrixSpace(GF(2), 4*npoints, (kX+kZ))(S);
end

"""
    computeScrewStabilisers(
      dim::Cart,
      defects::Vector{Tuple{Tuple{Int64, Int64}, Int64, LeftRight}};
      boundary::Vector{BoundaryType}=BoundaryType[]
    )

Computes stabilisers for a screw dislocation in the z direction, with periodic 
boundary conditions in the z direction. The defect is specified by:
```
    ((x1, x2), y, LeftRight)
```
where `(x1, x2)` specifies the x-positions of the two ends of the screw. `y` 
specifies the y-position of both screws. `LeftRight = Left` or `Right` indicates
whether the `x2` screw is left or right handed. 
                                                            
To put just one screw on the lattice, set `x1` or `x2` to be larger/smaller than the lattice dimension. 

Example: 
```
# Two left-handed screws:
computeScrewStabilisers(dim,
    Tuple{Tuple{Int64,Int64},Int64,LeftRight}[
        ((5,100), 4, Right),
        ((-100,4), 4, Left)
    ],
    boundary=[mType, eType]
);
```
"""
function computeScrewStabilisers(dim::Cart, 
    defects::Vector{Tuple{Tuple{Int64, Int64}, Int64, LeftRight}};
    boundary::Vector{BoundaryType}=BoundaryType[])
  npoints = prod(dim) |> Int;
  sdim = dim .+ 1;
  sdimx, sdimy, sdimz = sdim;

  # Periodic boundary conditions in the z direction
    # and boundaries on the x and y directions
  full = 1:prod(sdim);
  Xs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (z < sdimz) &&
      (mType ∈ boundary || ((1<y<sdimy) && (1<x<sdimx))) && 
      all(map(defects) do (posxs, posy, leftright)
            (x ∉ posxs || y != posy)
          end)
  end;
  Zs = filter(full) do loc 
    x,y,z = locToXYZ(loc, sdim...);
    (z < sdimz) && 
      (eType ∈ boundary || ((1<y<sdimy) && (1<x<sdimx))) && 
      all(map(defects) do (posxs, posy, leftright)
            (x ∉ posxs || y != posy)
          end)
  end;

  # Pad with an extra qubit on each face to construct the stabilisers on
  # the boundaries
  lx, ly, lz = dim .+ 2;
  kX, kZ = length(Xs), length(Zs);

  # Matrices to store the acted on states
  ρ0 = zeros(Int, lx, ly, lz); 

  # S matrix to store the stabilisers 
  S = zeros(Int, 4*npoints, kX+kZ);

  # Create X stabilisers
  for (index, loc) in enumerate(Xs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    if all(map(defects) do ((posx1, posx2), posy, _) 
                    (y != posy || !(posx1 <= x <= posx2))
                end) 
      # Regular bulk
      σ_indices = [
                   (0,0,0),
                   (1,1,0),
                   (1,0,1),
                   (0,1,1)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (0,1,0),
                   (0,0,1)];
    elseif any(map(defects) do ((posx1, posx2), posy, leftright) 
                (y == posy && (posx1 < x < posx2) && leftright == Left)
                end)
      # Slanted bulk up
      σ_indices = [
                   (0,0,0),
                   (1,1,1),
                   (1,0,1),
                   (0,1,2)];
      μ_indices = [
                   (0,0,0),
                   (1,0,0),
                   (0,1,1),
                   (0,0,1)];
    elseif any(map(defects) do ((posx1, posx2), posy, leftright) 
                (y == posy && (posx1 < x < posx2) && leftright == Right)
                end)
      # Slanted bulk down
      σ_indices = [
                   (0,0,1),
                   (1,1,0),
                   (1,0,2),
                   (0,1,1)];
      μ_indices = [
                   (0,0,1),
                   (1,0,1),
                   (0,1,0),
                   (0,0,2)];
    elseif any(map(defects) do (posx, posy, _) 
                    (y == posy && x ∈ posx) 
                end)
        @error "Shouldn't be here X (defect)"
    else 
        @error "Shouldn't be here X"
    end
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    σ_indices = map(σ_indices) do (x,y,z) 
        if z == 1
            z = sdimz;
        elseif z >= lz
            z = z - lz + 2;
        end
        (x,y,z)
    end;
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    μ_indices = map(μ_indices) do (x,y,z) 
        if z == 1
            z = sdimz;
        elseif z >= lz
            z = z - lz + 2;
        end
        (x,y,z)
    end;
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to S 
    S[1:2*npoints, index] = vcat(S_σ, S_μ); 
  end

  # Z stabilisers
  for (index, loc) in enumerate(Zs) 
    x, y, z = locToXYZ(loc, sdim...);
    
    # Apply the stabiliser
    ρ_μ, ρ_σ = copy(ρ0), copy(ρ0);
    if all(map(defects) do ((posx1, posx2), posy, _) 
                    (y != posy || !(posx1 <= x <= posx2))
                end) 
        # Bulk
      σ_indices = [
                   (1,1,0),
                   (1,0,1),
                   (1,1,1),
                   (0,1,1)];
      μ_indices = [
                   (1,0,0),
                   (0,1,0),
                   (0,0,1),
                   (1,1,1)];
    elseif any(map(defects) do ((posx1, posx2), posy, leftright) 
                (y == posy && (posx1 < x < posx2) && leftright == Left)
                end)
      # Slanted bulk
      σ_indices = [
                   (1,1,1),
                   (1,0,1),
                   (1,1,2),
                   (0,1,2)];
      μ_indices = [
                   (1,0,0),
                   (0,0,1),
                   (0,1,1),
                   (1,1,2)];
    elseif any(map(defects) do ((posx1, posx2), posy, leftright) 
                (y == posy && (posx1 < x < posx2) && leftright == Right)
                end)
      # Slanted bulk down
      σ_indices = [
                   (1,1,0),
                   (1,0,2),
                   (1,1,1),
                   (0,1,1)];
      μ_indices = [
                   (1,0,1),
                   (0,1,0),
                   (0,0,2),
                   (1,1,1)];
    elseif any(map(defects) do (posx, posy, _) 
                    (y == posy && x ∈ posx)
                end)
        @error "Shouldn't be here Z (defect)"
    else 
        @error "Shouldn't be here Z"
    end
    σ_indices = [(x,y,z) .+ ind for ind in σ_indices];
    σ_indices = map(σ_indices) do (x,y,z) 
        if z == 1
            z = sdimz;
        elseif z >= lz
            z = z - lz + 2;
        end
        (x,y,z)
    end;
    μ_indices = [(x,y,z) .+ ind for ind in μ_indices];
    μ_indices = map(μ_indices) do (x,y,z) 
        if z == 1
            z = sdimz;
        elseif z >= lz
            z = z - lz + 2;
        end
        (x,y,z)
    end;
    ρ_σ[CartesianIndex.(σ_indices)] .= 1;
    ρ_μ[CartesianIndex.(μ_indices)] .= 1;

    # Remove "outside" qubits and flatten 
    S_σ = ρ_σ[2:end-1, 2:end-1, 2:end-1][:];
    S_μ = ρ_μ[2:end-1, 2:end-1, 2:end-1][:];

    # Save to lambda 
    S[2*npoints+1:end, index + kX] = vcat(S_σ, S_μ); 
  end

  return MatrixSpace(GF(2), 4*npoints, (kX+kZ))(S);
end;

"""
    findLogicals(S; simplify=true, check=true)

Compute the qubits corresponding to logical operators, given a set of stabiliser
operators as the columns in S. 

Outputs: 
- `k`: The number of logical pairs
- `logicalsX`, `logicalsY`: 
    Each is a matrix with dimensions (u, v) where u is twice the number of qubits
    (half the dimension of S). Only k will be non-zero. Each of
    these columns is a logical operator. The ones indicate qubits acted on by
    that operator.
"""
function findLogicals(S; simplify=true, check=true)
  ma, mb = size(S);
  γ = zero_matrix(GF(2), ma, ma); 
  for i = 1:ma
    γ[i, Int(mod(i+(ma/2 - 1), ma)+1)] = 1;
  end
  ϵ = transpose(S) * γ;

  if check 
    @assert ϵ * S == 0 "Stabilisers do not all commute.";
  end

  # 1. Calculate an independent set of operators that commute with the
  # stabilisers 
  η = nullspace(ϵ)[2];

  # 2. Compute the anticommutation relation between them
  ξ = transpose(η) * γ * η; 
  # 3. Determine which combination of operators give an anticommuting pair
  S, T, U = snf_with_transform(ξ);
  # 4. Split up the resultant logicals into X and Z pairs
  logicalsX = (η * U * S)[1:Int(ma/2),:];
  logicalsZ = (η * transpose(T) * S)[Int(ma/2+1):end,:];

  # 5. Compute number of logicals
  k = rank(logicalsX);

  # 6. Simplify
  if simplify
    logicalsX = logicalsX[:, 1:k];
    logicalsZ = logicalsZ[:, 1:k];
  end

  return k, logicalsX, logicalsZ
end

"""
    findSubLogicals(func, S, dim::Cart)

Find logical operators in a given subset of the qubits in a lattice. 
`func` should be a function that accepts `(x,y,z)` and returns a boolean. `S` is
the stabilizer matrix, and `dim` are the dimensions of the lattice.
"""
function findSubLogicals(func, S, dim::Cart; lookup=1:prod(dim))
  npoints = length(lookup);
  qubits = filter(lookup) do loc 
    x,y,z = locToXYZ(loc, dim...);
    func(x,y,z)
  end;
  qubits = removeNothings(indexin(qubits, lookup));
  subset = qubitsToSubset(qubits, npoints);
  S1 = S[subset, :];

  # 1. Calculate an independent set of operators that commute with the
  # stabilisers 
  ηX = nullspace(transpose(half(S1, 2)))[2]; 
  ηZ = nullspace(transpose(half(S1)))[2];
  ηX = expand(ηX, subset, npoints);
  ηZ = expand(ηZ, subset, npoints);

  # 2. Find columns that are not in the column space of S
  logicalsX = vecQuotient(ηX, half(S));
  logicalsZ = vecQuotient(ηZ, half(S,2));

  # 3. Compute number of qubits
  kX = rank(logicalsX);
  kZ = rank(logicalsZ);
  k = (kX == kZ) ? kX : (kX, kZ);

  return k, logicalsX, logicalsZ 
end


"""
    getk(S)

Compute the number of encoded qubits, `k`, given a stabiliser matrix `S`.
"""
function getk(S)
  NX = nullspace(transpose(half(S,1)))[2];
  NZ = nullspace(transpose(half(S,2)))[2];
  rank(transpose(NX) * NZ)
end;
                                                        
"""
        plotLogical(L, dim, args...; x=1, y=2, jitter=0.5, kwargs...)
                                                            
Plots a logical operator, `L`, given lattice with dimension `dim`. `x,y` keyword
arguments refer to what lattice axis to put on the plotting axes. Jitter is applied
to each point to distinguish between the two single-qubit operators.
"""
function plotLogical(L, dim, args...; x=1, y=2, jitter=0.05, name = "X", lookup=1:prod(dim), kwargs...)
    points = findOnes(L);
    temp = qubitsToXYZ(points, dim...; lookup=lookup);
    second(x) = x[2];
    coords = second.(temp);
    qubits = map(first.(temp)) do group 
        group == 0 ? name*"I" : "I"*name
    end;
    shape = name == "X" ? Shape.utriangle : Shape;
    jit = map(first.(temp)) do group 
        group == 0 ? jitter : -jitter
    end;
    xs = getindex.(coords, x) .+ jit;
    ys = getindex.(coords, y) .+ jit;
    plot(x=xs, y=ys, color=qubits, shape=[shape], Geom.point,
        Guide.xlabel("$(('x','y','z')[x])"),
        Guide.ylabel("$(('x','y','z')[y])"),
        Guide.colorkey(title=""),
        Guide.shapekey(nothing),
        Coord.cartesian(xmin=1-0.1, xmax=dim[x]+0.1, 
            ymin=1-0.1, ymax=dim[y]+0.1),
        Guide.xticks(ticks=1:dim[x]),
        Guide.yticks(ticks=1:dim[y]),
        Scale.color_discrete(p -> [colorant"rgb(176, 83, 236)", colorant"rgb(126, 204, 95)"]),
        args...; kwargs...
    )
end;

# ---------------------------------------------------------------------- #
# HELPER FUNCTIONS
# ---------------------------------------------------------------------- #

ζ(L) = mod(L,1) == 0 ? 2^(findfirst(digits(Int(L), base=2) .== 1) - 1) : 0;
τ(L1, L2) = Int(min(ζ(L1/3), L2));

_hnf(X) = transpose(hnf(transpose(X)));
second(ls) = ls[2];

"""
    Γ(A), Γ(a::Int64, b::Int64)

Computes the gamma matrix to reverse the second and first half of 
the rows of `A`.
"""
function Γ(a::Int64, b::Int64)
  γ = zero_matrix(GF(2), a, a); 
  for i = 1:a
    γ[i, Int(mod(i+(a/2 - 1), a)+1)] = 1;
  end
  γ
end
function Γ(A)
  Γ(size(A)...);
end
                                                            
"""
    expand(A, indices::Vector{In64}, npoints::Int64})

Given a contracted matrix `A`, write it in expanded form.

# Examples
```
A0 = [1 2; 3 4; 5 6; 7 8];
indices = [1, 2];
A = A0[indices, :];
expand(A, indices, 1)
```
"""
function expand(A, indices::Vector{Int64}, npoints::Int64)
  M = zero_matrix(GF(2), 2*npoints, size(A, 2));
  for i in 1:Int(length(indices)/2)
    M[indices[i],:]  = A[i,:];
  end 
  M
end

"""
    intersectQubits(A, B, subsetA, subsetB)

Given two matrices `A` and `B`, contracted to only `subsetA` and `subsetB`
respectively, computes the intersection of `subsetA` and `subsetB` to return 
the joint elements of `A` and `B`.
"""
function intersectQubits(A, B, subsetA, subsetB)
  ia = findall(in(subsetB), subsetA);
  ib = findall(in(view(subsetA, ia)), subsetB);
  A[ia, :], B[ib, :], view(subsetA, ia)
end

"""
    qubitsToSubset(qubits, npoints::Int64)

Convert a vector of qubits into an array indexing the rows of S.
"""
function qubitsToSubset(qubits::Vector{Int64}, npoints::Int64)
  subset = [qubits; npoints .+ qubits];
  [subset; 2*npoints .+ subset]
end

"""
    setZero!(A, indices; not=false)

Set the rows of `A` corresponding to `indices` to `0`;
"""
function setZero!(A, indices; not=false)
  if not 
    indices = [ind for ind in 1:size(A,2) if !(ind ∈ indices)];
  end
  for ind in indices
    A[ind, :] = A[ind, :]*0;
  end 
end 

"""
    function augment(A, B)

Given two matrices \$A\$, \$B\$, return the augmented matrix 
    [A B] 

Fails if matrix height dimensions do not agree.
"""
function augment(A, B)
  ha, wa = size(A);
  hb, wb = size(B);
  @assert ha == hb "Heights are not identical"
  M = zero_matrix(GF(2), ha, (wa+wb));
  M[:, 1:wa] = A;
  M[:, (wa+1):(wa+wb)] = B;
  M
end

"""
    vecQuotient(A, B)

Given two matrices A (c x a) and B (c x b), compute a set of (c x 1) basis 
vectors that span the vector space \$im(A) \\ im(B)\$. 
"""
function vecQuotient(A, B)
  A = _hnf(A);
  B = _hnf(B); 
  a, b = size(A, 2), size(B, 2);

  M = augment(B, A);
  H = hnf(M);
  inds = getLeading(H); 
  filter!(x -> x > b, inds);

  M[:, inds]
end

"""
    getLeading(H)::Vector{Int64}

Return the indices of the columns corresponding to leading ones
in `H`. 

# Examples 
```
H = hnf(X);
indCols = H[:,getLeading(H)];
```
"""
function getLeading(H)::Vector{Int64}
  r = size(H, 1);
  results = [];
  for i=1:r
    value = findfirst(H[i,:] .== 1);
    if isnothing(value)
      continue 
    end 
    push!(results, something(value)[2]);
  end
  results
end

"""
    locToXYZ(loc::Int64, lx::Int64, ly::Int64, lz::Int64)

Determine the location of the stabiliser in (x,y,z) coordinates
anchored to the bottom corner
"""
function locToXYZ(loc::Int64, lx::Int64, ly::Int64, lz::Int64)
  x = mod(loc-1, lx)+1;
  y = mod(floor(Int16, (loc-1)/lx), ly)+1;
  z = mod(floor(Int16, (loc-1)/(lx*ly)), lz)+1;
  return x, y, z
end
                                                            
function qubitsToXYZ(qubits::Vector{Int64}, lx::Int64, ly::Int64, lz::Int64; lookup=1:(lx*ly*lz))
  npoints = length(lookup);
  groups = mod.(ceil.(qubits / npoints), 2) .- 1;
  qubits = mod.(qubits .- 1, npoints) .+ 1;
  locs = lookup[qubits];
  (collect ∘ zip)(groups, locToXYZ.(locs, lx, ly, lz))
end;

"""
    xyzToLoc(xyz::Tuple{Int64, Int64, Int64}, lx::Int64, ly::Int64, lz::Int64)

Determine the index of a stabiliser or qubit in an `lx × ly × lz` 
lattice, given a cartesian coordinate (x,y,z). This is the inverse
of `locToXYZ()`.

# Example 
```
xyz = locToXYZ(5, 3, 3, 3);
loc = xyzToLoc(xyz, 3, 3, 3); # == 5
```
"""
function xyzToLoc(xyz::Tuple{Int64, Int64, Int64}, lx::Int64, ly::Int64, lz::Int64)
  x, y, z = xyz; 
  loc = x + (y-1) * lx + (z-1) * (lx * ly);
  return loc 
end 

"""
    qubitsToVector(σ, μ, npoints; is_x=true)

Construct a vector representing an operator with given qubits with indices in σ
and μ. `npoints` is the number of total points in the lattice (half the number
of qubits)

# Example
```
ρ = qubitsToVector([1,2], [1], 3);
# ρ = [1,1,0,1,0,0,0,0,0,0,0,0]
```
"""
function qubitsToVector(σ, μ, npoints; is_x=true)
  ρ = zero_matrix(GF(2), npoints*4, 1);
  shift = is_x ? 0 : 2*npoints; 
  for i in σ
    ρ[i+shift, 1] = 1;
  end
  for i in μ
    ρ[npoints+shift+i, 1] = 1;
  end
  return ρ
end

"""
   stabsToVector(X, Z, dim)

Construct a vector representing an operator comprised of the given X and Z 
stabilisers. Useful for constructing strings of plaquettes, for example. 

`Xs` and `Zs` should be vectors of indices corresponding to the location of 
the stabilisers to be applied. Use `xyzToLoc()` to create these from coordinates.
"""
function stabsToVector(Xs::Vector{Int64}, Zs::Vector{Int64}, dim)
  τ = computeStabilisers(Xs, Zs, dim);
  # Compress into one column
  τ = foldl((x,y) -> τ[:,y] + x, 1:ncols(τ); init=0);
  return τ
end
"""
    stabsToVector(Xs::Vector{Tuple{Int64,Int64,Int64}},
      Zs::Vector{Tuple{Int64,Int64,Int64}}, dim)
"""
function stabsToVector(Xs::Vector{Cart}, 
    Zs::Vector{Cart}, dim::Cart)
  Xs = xyzToLoc.(Xs, (dim .+ 1)...);
  Zs = xyzToLoc.(Zs, (dim .+ 1)...);
  stabsToVector(Xs, Zs, dim);
end
function stabsToVector(Xs::Vector{Cart}, Zs::Vector{Cart}, dim::Int64) 
  stabsToVector(Xs, Zs, (dim, dim, dim))
end

"""
    isStabiliser(S, v)

Tests if an operator `v` is contained within the stabiliser group `S`, by seeing 
if the rank of `S` changes when augmented. 
"""
function isStabiliser(S, v)
  A = augment(S, v);
  return rank(A) == rank(S)
end

"""
    findOnes(arr)
Finds the indices of elements in `arr` that `== 1`.
"""
function findOnes(arr) 
  return first.(Tuple.(findall(arr .== 1)))
end

"""
    half(arr, which=1)
Get the first (or second) half of an even-length array.
"""
function half(arr, which=1)
  n = Int(size(arr, 1) / 2);
  if which == 1 
    return arr[1:n, :]
  else 
    return arr[(n+1):end, :];
  end
end

"""
    findExciteds(stabs, S, v, dim)

Output the coordinates of the excited stabilisers given an operator, `v`.
`dim` is the dimensions of the qubit lattice - either a single number of a
length-3 vector.
"""
function findExciteds(S, v, stabX, stabZ, dim)
  if length(dim) == 1
    dim = (dim, dim, dim);
  end 
  # Compute Z excitations and X excitations as indices
  excsZ = stabZ[transpose(half(S, 2)) * half(v,1) |> findOnes];
  excsX = stabX[transpose(half(S, 1)) * half(v,2) |> findOnes];
  # Convert to coordinates
  return locToXYZ.([excsX; excsZ], (dim.+1)...)
end

"""
    classify(S, v, stabs, dim)

Classify an operator `v` as a stabiliser, logical, or excitation.
"""
function classify(S, v, stabX, stabZ, dim::Cart)
  stab = isStabiliser(S, v);
  if stab 
    return "stabiliser"
  end 
  excit = length(findExciteds(S, v, stabX, stabZ, dim)) != 0;
  if excit 
    return "excitation"
  end 
  return "logical"
end 
function classify(S, v, stabX, stabZ, dim::Int64)
  classify(S, v, stabX, stabZ, (dim, dim, dim))
end
                                                            
"""
    checkCommutes(S)

Check whether a stabiliser matrix consists of commuting operators
"""
function checkCommutes(S)
    rank(transpose(half(S)) * half(S,2)) == 0
end;
                                                            
function removeNothings(vec::Vector{Union{Nothing, Int64}})::Vector{Int64}
    [x for x in vec if !isnothing(x)]
end;

"""
    getQubitLookup(dim::Cart, vacancies::Vector{Vacancy})

Specifies a lookup list for a lattice with vacancies, where
```
    vacancies = Vacancy[
        ((1,1,1), (1,1,1), mType)
    ];
```
for example. 
"""
function getQubitLookup(dim::Cart, vacancies::Vector{Vacancy})
  npoints = prod(dim) |> Int;
  full = 1:npoints;
  qubitLookup = filter(full) do loc
    x,y,z = locToXYZ(loc, dim...);
    !any(map(vacancies) do (pos, size, btype)
      all(pos .<= (x,y,z) .< (pos .+ size))
    end)
  end;
  return qubitLookup;
end;