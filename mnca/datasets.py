from torch_geometric.data import Data


class Icosahedron(Data):
    def __init__(self):
        phi = (1 + 5**0.5) / 2
        # Define the vertices of the icosahedron
        vertices = [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ]

        # Define the edges of the icosahedron
        edges = [
            [0, 1],
            [0, 4],
            [0, 5],
            [0, 8],
            [0, 10],
            [1, 6],
            [1, 7],
            [1, 8],
            [1, 10],
            [1, 11],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 9],
            [2, 11],
            [3, 6],
            [3, 7],
            [3, 9],
            [3, 10],
            [3, 11],
            [4, 5],
            [4, 8],
            [4, 9],
            [5, 9],
            [5, 10],
            [6, 7],
            [6, 8],
            [6, 9],
            [7, 8],
            [7, 10],
        ]

        # Create the Data object
        super().__init__()

        # Set the attributes of the Data object
        self.pos = torch.tensor(vertices, dtype=torch.float)
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
