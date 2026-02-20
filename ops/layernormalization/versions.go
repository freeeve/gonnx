package layernormalization

import "github.com/advancedclimatesystems/gonnx/ops"

var layerNormVersions = ops.OperatorVersions{
	17: ops.NewOperatorConstructor(newLayerNormalization, 17, layerNormTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return layerNormVersions
}
