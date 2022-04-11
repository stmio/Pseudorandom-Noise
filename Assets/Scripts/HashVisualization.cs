using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;

public class HashVisualization : MonoBehaviour {

    static int
        hashesId = Shader.PropertyToID("_Hashes"),
        configId = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    NativeArray<uint> hashes;

    ComputeBuffer hashesBuffer;

    MaterialPropertyBlock propertyBlock;

    [SerializeField, Range(-2f, 2f)]
    float verticalOffset = 1f;
        
    [SerializeField]
    int seed;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS {
        scale = 8f
    };

    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor {

        [WriteOnly]
        public NativeArray<uint> hashes;

        public int resolution;

        public float invResolution;

        public SmallXXHash hash;

        public float3x4 domainTRS;

        public void Execute(int i) {
            float vf = (int)floor(invResolution * i + 0.00001f);
            float uf = invResolution * (i - resolution * vf + 0.5f) - 0.5f;
            vf = invResolution * (vf + 0.5f) - 0.5f;

            float3 p = mul(domainTRS, float4(uf, 0f, vf, 1f));

            int u = (int)floor(p.x);
            int v = (int)floor(p.y);
            int w = (int)floor(p.z);

            hashes[i] = hash.Eat(u).Eat(v).Eat(w);
        }
    }

    void OnEnable() {
        int length = resolution * resolution;
        hashes = new NativeArray<uint>(length, Allocator.Persistent);
        hashesBuffer = new ComputeBuffer(length, 4);

        new HashJob {
            hashes = hashes,
            resolution = resolution,
            invResolution = 1f / resolution,
            hash = SmallXXHash.Seed(seed),
            domainTRS = domain.Matrix
        }.ScheduleParallel(hashes.Length, resolution, default).Complete();
    
        hashesBuffer.SetData(hashes);
    
        propertyBlock ??= new MaterialPropertyBlock();
        propertyBlock.SetBuffer(hashesId, hashesBuffer);
        propertyBlock.SetVector(configId, new Vector4(
            resolution, 1f / resolution, verticalOffset / resolution
        ));
    }

    void OnDisable() {
        hashes.Dispose();
        hashesBuffer.Release();
        hashesBuffer = null;
    }

    void OnValidate() {
        if (hashesBuffer != null && enabled) {
            OnDisable();
            OnEnable();
        }
    }

    void Update() {
        Graphics.DrawMeshInstancedProcedural(
            instanceMesh, 0, material, new Bounds(Vector3.zero, Vector3.one),
            hashes.Length, propertyBlock
        );
    }
}
