// Pseudocode
void OnMessage(string json){
    float[] coeffs = JsonHelper.FromJson<float>(json);
    for(int i=0;i<coeffs.Length;i++)
        skinnedMeshRenderer.SetBlendShapeWeight(i, coeffs[i]*100f);
}
