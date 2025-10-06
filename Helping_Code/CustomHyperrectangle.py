from typing import Dict, Tuple, Optional, List
import numpy as np
import itertools

def vertices_from_bounds_dict(
    bounds_by_idx: Dict[int, Tuple[float, float]],
    feature_order: Optional[List[int]] = None
) -> np.ndarray:
    """
    Convert per-dimension bounds (dict of {dim_index: (min, max)}) into the
    vertices of the hyperrectangle as a (2^d, d) numpy array.

    Args:
        bounds_by_idx: {dim_index: (min, max)} for each bounded dimension.
        feature_order: optional explicit order of dimensions; if None, uses sorted keys.

    Returns:
        vertices: np.ndarray of shape (2^d, d)
    """
    if not isinstance(bounds_by_idx, dict) or len(bounds_by_idx) == 0:
        raise ValueError("bounds_by_idx must be a non-empty dict {dim: (min, max)}.")

    # Determine consistent dimension order
    dims = feature_order if feature_order is not None else sorted(bounds_by_idx.keys())

    # Validate and collect (min, max) for each dim
    pairs = []
    for d in dims:
        if d not in bounds_by_idx:
            raise ValueError(f"Dimension {d} missing in bounds_by_idx.")
        mn, mx = bounds_by_idx[d]
        if mn > mx:
            raise ValueError(f"Invalid bounds for dim {d}: min {mn} > max {mx}.")
        pairs.append((float(mn), float(mx)))

    # Cartesian product over (min,max) for each dimension -> all corners
    # itertools.product yields tuples length d; stack to (2^d, d)
    vertices = np.array(list(itertools.product(*pairs)), dtype=float)
    return vertices


def minimize_hyperrectangle_distance_dual(V1, V2):
        """Compute minimum Euclidean distance between two hyperrectangles using dual convex combination."""
        V1 = np.array(V1)
        V2 = np.array(V2)
        n, d = V1.shape
        m, _ = V2.shape

        beta0 = np.ones(n) / n
        alpha0 = np.ones(m) / m
        z0 = np.concatenate([beta0, alpha0])

        def objective(z):
            return np.linalg.norm(z[:n].dot(V1) - z[n:].dot(V2))**2

        bounds = [(0, 1)] * (n + m)
        cons = [{'type': 'eq', 'fun': lambda z: np.sum(z[:n]) - 1},
                {'type': 'eq', 'fun': lambda z: np.sum(z[n:]) - 1}]

        result = minimize(objective, z0, bounds=bounds, constraints=cons, method='SLSQP', options={
            'ftol': 1e-8, 'disp': False, 'maxiter': 1000
        })

        beta_opt = result.x[:n]
        alpha_opt = result.x[n:]
        X_opt = beta_opt.dot(V1)
        Xp_opt = alpha_opt.dot(V2)
        min_distance = np.linalg.norm(X_opt - Xp_opt)

        return X_opt, Xp_opt, min_distance







import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize



class CustomHyperrectangle:

    def __init__(self):
        pass

    def getVerticesFromBounds(self, bounds):
        """Compute vertices from min/max bounds for each hyperrectangle."""
        if not isinstance(bounds, np.ndarray) or bounds.size == 0:
            raise ValueError("Input must be a non-empty 2D array.")

        numHyperrectangles = bounds.shape[0]
        numDimensions = bounds.shape[1] // 2

        if bounds.shape[1] % 2 != 0:
            raise ValueError("Each hyperrectangle must have min and max bounds for each dimension.")

        vertices_list = []

        for i in range(numHyperrectangles):
            hr_bounds = bounds[i, :]
            vertexList = []

            for j in range(2**numDimensions):
                vertex = np.zeros(numDimensions)
                for k in range(numDimensions):
                    if j & (1 << k):
                        vertex[k] = hr_bounds[2*k + 1]
                    else:
                        vertex[k] = hr_bounds[2*k]
                vertexList.append(vertex)
            vertices_list.append(np.array(vertexList))

        return vertices_list

    def getBoundsFromVertices(self, vertices):
        """Compute min/max bounds from hyperrectangle vertices."""
        if not isinstance(vertices, np.ndarray) or vertices.size == 0:
            raise ValueError("Input must be a non-empty 2D matrix.")

        minBounds = np.min(vertices, axis=0)
        maxBounds = np.max(vertices, axis=0)

        bounds = np.zeros(2 * vertices.shape[1])
        bounds[0::2] = minBounds
        bounds[1::2] = maxBounds

        return bounds

    # def minimize_hyperrectangle_distance_dual(self, V1, V2):
    #     """Compute minimum Euclidean distance between two hyperrectangles using dual convex combination."""
    #     V1 = np.array(V1)
    #     V2 = np.array(V2)
    #     n, d = V1.shape
    #     m, _ = V2.shape

    #     beta0 = np.ones(n) / n
    #     alpha0 = np.ones(m) / m
    #     z0 = np.concatenate([beta0, alpha0])

    #     def objective(z):
    #         return np.linalg.norm(z[:n].dot(V1) - z[n:].dot(V2))**2

    #     bounds = [(0, 1)] * (n + m)
    #     cons = [{'type': 'eq', 'fun': lambda z: np.sum(z[:n]) - 1},
    #             {'type': 'eq', 'fun': lambda z: np.sum(z[n:]) - 1}]

    #     result = minimize(objective, z0, bounds=bounds, constraints=cons, method='SLSQP', options={
    #         'ftol': 1e-8, 'disp': False, 'maxiter': 1000
    #     })

    #     beta_opt = result.x[:n]
    #     alpha_opt = result.x[n:]
    #     X_opt = beta_opt.dot(V1)
    #     Xp_opt = alpha_opt.dot(V2)
    #     min_distance = np.linalg.norm(X_opt - Xp_opt)

    #     return X_opt, Xp_opt, min_distance

    def findSmallestHyperrectangleAndEdge(self, hyperrectangles):
        """Find the smallest hyperrectangle and its smallest edge."""
        volumes = []

        for verts in hyperrectangles:
            verts = np.array(verts)
            if verts.shape[0] < verts.shape[1]:
                verts = verts.T
            try:
                hull = ConvexHull(verts)
                volumes.append(self.volumeConvexHull(verts, hull.simplices))
            except:
                volumes.append(np.inf)

        min_idx = int(np.argmin(volumes))
        smallestHR = hyperrectangles[min_idx]

        numVertices = smallestHR.shape[0]
        smallestEdgeLength = np.inf
        edgeVertices = None
        hasZeroEdge = False

        for i in range(numVertices):
            for j in range(i+1, numVertices):
                edgeLength = np.linalg.norm(smallestHR[i] - smallestHR[j])
                if 0 < edgeLength < smallestEdgeLength:
                    smallestEdgeLength = edgeLength
                    edgeVertices = np.vstack([smallestHR[i], smallestHR[j]])
                elif edgeLength == 0:
                    hasZeroEdge = True

        if hasZeroEdge and smallestEdgeLength == np.inf:
            print("Warning: Only zero-length edges were found.")
            smallestEdgeLength = 0
            edgeVertices = None

        return smallestHR, smallestEdgeLength, edgeVertices

    @staticmethod
    def volumeConvexHull(vertices, simplices):
        """Compute volume of convex hull."""
        volume = 0
        for simplex in simplices:
            pts = vertices[simplex]
            mat = np.hstack([pts, np.ones((pts.shape[0], 1))])
            volume += abs(np.linalg.det(mat)) / np.math.factorial(pts.shape[0] - 1)
        return volume
