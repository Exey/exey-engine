//! Small directed graph + Kahn's topological sort.
//!
//! Mirrors AS3's `Graph` + `GraphTopologicalSorter` from
//! `ragcat.engine.common.graph`. Used by the iso-rectangle sorter to
//! turn pairwise "A renders before B" constraints into a globally
//! consistent draw order.
//!
//! ## Cycle handling
//!
//! Topological sort fails on cycles. With well-formed iso geometry,
//! cycles shouldn't arise (no three sprites can pairwise occlude each
//! other). But pathological inputs are constructible, and asserting
//! crashes is worse than rendering slightly-wrong. The sorter detects
//! cycles, logs a warning, and appends the un-sorted nodes in original
//! index order. That gives graceful degradation.

/// Directed graph over `0..n` nodes. Stored as adjacency lists; small
/// enough that we don't bother with a CSR representation.
pub struct Graph {
    pub n: usize,
    /// `edges[from]` is the list of `to` nodes for edges `from → to`.
    pub edges: Vec<Vec<u32>>,
}

impl Graph {
    pub fn with_nodes(n: usize) -> Self {
        Self { n, edges: vec![Vec::new(); n] }
    }

    /// Add a directed edge `from → to`. We don't deduplicate; multiple
    /// edges between the same pair count as one constraint for the
    /// topological sort (Kahn's only checks in-degree, and we count
    /// down to zero). The sorter calls `add_edge` from sweep neighbours;
    /// duplicates would only arise from buggy input — log if seen.
    pub fn add_edge(&mut self, from: u32, to: u32) {
        debug_assert!(
            (from as usize) < self.n && (to as usize) < self.n,
            "edge {from}→{to} out of bounds (n={})", self.n
        );
        self.edges[from as usize].push(to);
    }

    pub fn edge_count(&self) -> usize {
        self.edges.iter().map(Vec::len).sum()
    }
}

/// Kahn's algorithm. Returns a topological ordering: a permutation of
/// `0..n` such that for every edge `a → b`, `a` appears before `b`.
/// On a cyclic graph the result is "best effort": as many nodes as
/// possible in valid order, then the remaining (cyclic) nodes in their
/// original index order, with a warning logged.
pub fn topological_sort(graph: &Graph) -> Vec<u32> {
    let n = graph.n;
    let mut in_degree = vec![0u32; n];
    for from in 0..n {
        for &to in &graph.edges[from] {
            in_degree[to as usize] += 1;
        }
    }

    // Initialise the queue with every node of in-degree 0. We use a
    // simple Vec as a FIFO; for `n` in the thousands the cost is
    // dominated by edge traversal anyway.
    let mut queue: Vec<u32> = (0..n as u32)
        .filter(|&i| in_degree[i as usize] == 0)
        .collect();
    let mut result = Vec::with_capacity(n);
    let mut head = 0usize;

    while head < queue.len() {
        let node = queue[head];
        head += 1;
        result.push(node);
        for &to in &graph.edges[node as usize] {
            let d = &mut in_degree[to as usize];
            *d -= 1;
            if *d == 0 {
                queue.push(to);
            }
        }
    }

    if result.len() < n {
        // Cycle detected: append remaining nodes in original index order.
        // Log a single warning per sort call so we don't flood the console.
        let missing = n - result.len();
        log::warn!(
            "topological_sort: cycle detected, {missing} of {n} nodes \
             could not be ordered; appending them in input order"
        );
        for i in 0..n as u32 {
            if in_degree[i as usize] > 0 {
                result.push(i);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = Graph::with_nodes(0);
        assert_eq!(topological_sort(&g), Vec::<u32>::new());
    }

    #[test]
    fn no_edges() {
        let g = Graph::with_nodes(3);
        let r = topological_sort(&g);
        assert_eq!(r.len(), 3);
        // All three nodes appear, exact order is arbitrary but
        // deterministic for the queue init.
    }

    #[test]
    fn chain() {
        // 0 → 1 → 2: must come out as [0, 1, 2].
        let mut g = Graph::with_nodes(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert_eq!(topological_sort(&g), vec![0, 1, 2]);
    }

    #[test]
    fn diamond() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3
        let mut g = Graph::with_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 3);
        let r = topological_sort(&g);
        // 0 first, 3 last, 1 and 2 in between in either order.
        assert_eq!(r[0], 0);
        assert_eq!(r[3], 3);
    }

    #[test]
    fn cycle_falls_back_gracefully() {
        // 0 → 1 → 2 → 0
        let mut g = Graph::with_nodes(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        let r = topological_sort(&g);
        assert_eq!(r.len(), 3); // all nodes appear
    }
}
