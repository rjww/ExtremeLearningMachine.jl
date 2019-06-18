# Returns a partitioning of `range` into sub-ranges of length `partition_size`.
# If `range` is not divisible by `partition_size`, the last sub-range will be
# of length `range` mod `partition_size`.
function partition_range(range::UnitRange, partition_size::Int)
    n_partitions = ceil(Int, last(range) / partition_size)
    partitions = Vector{UnitRange{Int}}(undef, n_partitions)

    for p in 1:n_partitions
        l::Int = first(range) + (p-1) * partition_size
        r::Int = min(l - 1 + partition_size, last(range))
        partitions[p] = l:r
    end

    return partitions
end
