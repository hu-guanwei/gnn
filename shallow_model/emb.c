#include <stdio.h>
#include <stdlib.h>

const int num_nodes = 5;
const int dim_emb = 2;

#define elem_of(mat, i, j) mat[(i) * (dim_emb) + (j)]

void display_mat(float *mat, int n_rows, int n_cols) {
    printf("[\n");
    for (int i = 0; i < n_rows; i++) {
        printf("[ ");
        for (int j = 0; j < n_cols; j++) {
            printf("%.3f, ", elem_of(mat, i, j));
        }
        printf("],\n");
    }
    printf("],\n");
}

void rand_init_emb(float *emb) {
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < dim_emb; j++) {
            elem_of(emb, i, j) = (float)rand() / (0.0 + RAND_MAX);
        }
    }
}

float inner_pord(float *emb, int i, int j) {
    float prod = 0;
    for (int k = 0; k < dim_emb; k++) {
        prod += elem_of(emb, i, k) * elem_of(emb, j, k);
    }
    return prod;
}

void update(float *emb, int i, int j, int has_edge, float lr) {
    // if node i and node j are connected by an edge
    // then embedding vector of node i should turn towards embedding vector and node j, same for node j
    // ;
    // if they are not connected by an edge
    // thery turn towards the opposite direction of the other

    float prod = inner_pord(emb, i, j);
    for (int k = 0; k < dim_emb; k++) {
        float tmp = elem_of(emb, i, k);
        elem_of(emb, i, k) += (has_edge > 0 ? 1: -1) * lr * elem_of(emb, j, k);
        elem_of(emb, j, k) += (has_edge > 0 ? 1: -1) * lr * tmp;
    }
}


int main() {
    float *emb = malloc(num_nodes * dim_emb * sizeof(float));
    rand_init_emb(emb);
    int edge_index[][2] = {
        {0, 1},
        {1, 0},
        {0, 2},
        {2, 0},
        {1, 2},
        {2, 1},

        {3, 4},
        {4, 3}
    };
    int non_edges[][2] = {
        {0, 3}, {3, 0},
        {0, 4}, {4, 0},
        {1, 3}, {3, 1},
        {1, 4}, {4, 1},
        {2, 3}, {3, 2},
        {2, 4}, {4, 2},
    };
    for (int i = 0; i < 400; i++) {
        int j = i % 8;
        int k = i % 12;
        update(emb, edge_index[j][0], edge_index[j][1], 1, 0.01);
        update(emb, non_edges[k][0], non_edges[k][1], 0, 0.01);
    }
    display_mat(emb, num_nodes, dim_emb);
    free(emb);
    return 0;
}

