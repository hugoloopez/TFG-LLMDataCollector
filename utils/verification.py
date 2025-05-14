import psycopg2


def verificar_registro_completo(email, expected_data, db_uri):
    try:
        with psycopg2.connect(db_uri) as conn:
            with conn.cursor() as cur:

                cur.execute("SELECT cliente_id, nombre, apellidos, correo, telefono, dni FROM clientes WHERE correo = %s", (email,))
                cliente = cur.fetchone()
                if not cliente:
                    return False, "Cliente no registrado"
                cliente_id = cliente[0]
                cliente_ok = all(str(cliente[i+1]).strip() == str(expected_data["cliente"][key]).strip() for i, key in enumerate(["nombre", "apellidos", "correo", "telefono", "dni"]))

                cur.execute("SELECT calle, numero, ciudad, pais, codigo_postal FROM direcciones WHERE cliente_id = %s", (cliente_id,))
                direccion = cur.fetchone()
                if not direccion:
                    return False, "Registro incompleto: direccion"
                direccion_ok = all(str(direccion[i]).strip() == str(val).strip() for i, val in enumerate(expected_data["direccion"].values()))

                cur.execute("SELECT numero_tarjeta, titular_tarjeta, fecha_expiracion, cvv FROM metodos_pago WHERE cliente_id = %s", (cliente_id,))
                pago = cur.fetchone()
                if not pago:
                    return False, "Registro incompleto: pago"
                pago_ok = all(str(pago[i]).strip() == str(val).strip() for i, val in enumerate(expected_data["pago"].values()))

                cur.execute("SELECT plan_id, fecha_inicio, activo FROM clientes_planes WHERE cliente_id = %s", (cliente_id,))
                plan = cur.fetchone()
                if not plan:
                    return False, "Registro incompleto: plan"
                plan_ok = plan[2] is True

        if all([cliente_ok, direccion_ok, pago_ok, plan_ok]):
            return True, "Registro completo"
        else:
            errores = []
            if not cliente_ok:
                errores.append("cliente")
            if not direccion_ok:
                errores.append("direccion")
            if not pago_ok:
                errores.append("pago")
            if not plan_ok:
                errores.append("plan")
            return False, f"Registro incompleto: {', '.join(errores)}"

    except Exception as e:
        return False, f"Error al verificar: {e}"
